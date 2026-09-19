from __future__ import annotations

"""One-slot exact token-id continuation on top of ChunkCache.

Used when radix is off (SM70 CSA2 state is not in the tree) but a single
full-history chat should not re-prefill from position 0 every HTTP turn.

Hit only if ``new_ids[:len(last_ids)] == last_ids``. A shorter or different
prefix is a miss: drop the pin (CSA2 at pos 0 would already be stale) and
full-prefill. No ``session_id``. Do not wrap when streaming-session is on.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch

from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


logger = logging.getLogger(__name__)


class _VirtualNode:
    """Sentinel for inc/dec_lock_ref so ChunkCache's no-op lock stays no-op."""


@dataclass
class _Slot:
    virtual_node: _VirtualNode = field(default_factory=_VirtualNode)
    kv: ReqKvInfo = field(default_factory=ReqKvInfo)

    def save_from_req(self, req: Req, is_first: bool) -> None:
        kv = req.detach_kv()
        if is_first:
            self.kv = kv
        else:
            assert kv is self.kv

    def restore_to_req(self, req: Req) -> None:
        req.kv = self.kv


def _finished_token_ids(req: Req) -> list[int]:
    out = req.output_ids_through_stop
    return list(req.origin_input_ids) + list(out)


def _is_exact_continuation(new_ids: Sequence[int], last_ids: tuple[int, ...]) -> bool:
    n = len(last_ids)
    if n == 0 or len(new_ids) < n:
        return False
    return tuple(new_ids[:n]) == last_ids


class StickyLastSequenceCache(BasePrefixCache):
    """Pin the last finished request's KV; reuse only on exact continuation."""

    def __init__(self, inner: BasePrefixCache):
        self.inner = inner
        self._slot: Optional[_Slot] = None
        self._last_ids: Optional[tuple[int, ...]] = None
        self._last_extra_key: Optional[str] = None
        self._last_cache_salt: Optional[str] = None

    @property
    def req_to_token_pool(self):
        return self.inner.req_to_token_pool

    @req_to_token_pool.setter
    def req_to_token_pool(self, value):
        self.inner.req_to_token_pool = value

    @property
    def token_to_kv_pool_allocator(self):
        return self.inner.token_to_kv_pool_allocator

    @token_to_kv_pool_allocator.setter
    def token_to_kv_pool_allocator(self, value):
        self.inner.token_to_kv_pool_allocator = value

    @property
    def page_size(self):
        return self.inner.page_size

    @page_size.setter
    def page_size(self, value):
        self.inner.page_size = value

    @property
    def disable(self):
        return self.inner.disable

    @disable.setter
    def disable(self, value):
        self.inner.disable = value

    def is_chunk_cache(self) -> bool:
        return True

    def supports_streaming_session(self) -> bool:
        return False

    def reset(self) -> None:
        self._drop_slot("reset")
        self.inner.reset()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        hit = self._try_hit(params)
        if hit is not None:
            return hit
        return self.inner.match_prefix(params)

    def _try_hit(self, params: MatchPrefixParams) -> Optional[MatchResult]:
        if self._slot is None or self._last_ids is None:
            return None
        if not self._slot.kv.holds_kv:
            self._clear_pin()
            return None

        key = params.key
        extra_ok = key.extra_key == self._last_extra_key
        salt_ok = (key.cache_salt or None) == self._last_cache_salt
        new_ids = key.raw_token_ids()
        if not (extra_ok and salt_ok and _is_exact_continuation(new_ids, self._last_ids)):
            logger.info(
                "sticky last-seq miss (had %d tokens, new %d); drop pin, full prefill",
                len(self._last_ids),
                len(new_ids),
            )
            self._drop_slot("miss")
            return None

        req = params.req
        if req is None:
            self._drop_slot("miss-no-req")
            return None

        slot = self._slot
        slot.restore_to_req(req)
        prefix_len = len(self._last_ids)
        self._free_tail(req.kv, prefix_len)

        device_indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :prefix_len
        ].to(dtype=torch.int64)

        logger.info(
            "sticky last-seq hit prefix_len=%d new_len=%d",
            prefix_len,
            len(new_ids),
        )
        return MatchResult(
            device_indices=device_indices,
            last_device_node=slot.virtual_node,
            last_host_node=slot.virtual_node,
            best_match_node=slot.virtual_node,
            cache_protected_len=0,
        )

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        if isinstance(req.finished_reason, FINISH_ABORT):
            if self._slot is not None and req.kv is self._slot.kv:
                self._clear_pin()
            else:
                self._drop_slot("abort")
            self.inner.cache_finished_req(req, is_insert=is_insert, **kwargs)
            return

        ids = _finished_token_ids(req)
        finished_len = (
            req.finished_len if req.finished_len is not None else len(req.output_ids)
        )
        self._trim_overshoot(req, finished_len)
        ids = list(req.origin_input_ids) + list(req.output_ids[:finished_len])

        is_first = self._slot is None
        if is_first:
            self._slot = _Slot()
        self._slot.save_from_req(req, is_first=is_first)
        self._slot.kv.kv_committed_len = min(len(ids), self._slot.kv.kv_allocated_len)
        self._slot.kv.cache_protected_len = 0
        self._last_ids = tuple(ids)
        self._last_extra_key = getattr(req, "extra_key", None)
        self._last_cache_salt = getattr(req, "cache_salt", None) or None
        logger.info("sticky last-seq pin %d tokens", len(self._last_ids))

    def cache_unfinished_req(self, req: Req, **kwargs):
        self.inner.cache_unfinished_req(req, **kwargs)

    def insert(self, *args, **kwargs):
        return self.inner.insert(*args, **kwargs)

    def evict(self, params: EvictParams) -> EvictResult:
        return self.inner.evict(params)

    def evict_for_alloc(self, params: EvictParams) -> EvictResult:
        return self.inner.evict_for_alloc(params)

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        if isinstance(node, _VirtualNode):
            return IncLockRefResult()
        return self.inner.inc_lock_ref(node)

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if isinstance(node, _VirtualNode):
            return DecLockRefResult()
        return self.inner.dec_lock_ref(node, params)

    def _pin_idle(self, active_pool_idxs: Optional[set]) -> bool:
        slot = self._slot
        if slot is None or not slot.kv.holds_kv:
            return False
        if active_pool_idxs is not None and slot.kv.req_pool_idx in active_pool_idxs:
            return False
        return True

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        # ChunkCache has no tree-protected prefix; the whole pin is uncached.
        if not self._pin_idle(active_pool_idxs):
            return 0
        return ceil_align(self._slot.kv.kv_allocated_len, self.page_size)

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session_held_tokens(active_pool_idxs)

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        if not self._pin_idle(active_pool_idxs):
            return 0
        allocated = ceil_align(self._slot.kv.kv_allocated_len, self.page_size)
        return allocated - self._slot.kv.swa_evicted_seqlen

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:
        # Counted as allocatable in Scheduler.get_num_allocatable_reqs.
        return int(self._pin_idle(active_pool_idxs))

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def protected_size(self):
        return self.inner.protected_size()

    def evictable_size(self):
        return self.inner.evictable_size()

    def full_evictable_size(self):
        return self.inner.full_evictable_size()

    def swa_evictable_size(self):
        return self.inner.swa_evictable_size()

    def full_protected_size(self):
        return self.inner.full_protected_size()

    def swa_protected_size(self):
        return self.inner.swa_protected_size()

    def total_size(self):
        return self.inner.total_size()

    def pretty_print(self):
        n = 0 if self._last_ids is None else len(self._last_ids)
        return f"StickyLastSequence(pinned={n})\n{self.inner.pretty_print()}"

    def init_load_back(self, params: InitLoadBackParams):
        return self.inner.init_load_back(params)

    def pop_prefetch_loaded_span(self, req_id: str) -> tuple[int, Optional[int]]:
        return self.inner.pop_prefetch_loaded_span(req_id)

    def finish_storage_prefetch_admission(
        self, req_id: str, fulfilled_tokens: int, reason: Optional[str]
    ) -> None:
        self.inner.finish_storage_prefetch_admission(req_id, fulfilled_tokens, reason)

    def discard_storage_prefetch_accounting(self, req_id: str) -> None:
        self.inner.discard_storage_prefetch_accounting(req_id)

    def ready_to_load_host_cache(self):
        return self.inner.ready_to_load_host_cache()

    def check_hicache_events(self):
        return self.inner.check_hicache_events()

    def take_events(self):
        return self.inner.take_events()

    def supports_swa(self):
        return self.inner.supports_swa()

    def supports_mamba(self):
        return self.inner.supports_mamba()

    def available_and_evictable_str(self):
        return self.inner.available_and_evictable_str()

    def init_metrics_collector(self):
        return self.inner.init_metrics_collector()

    def sanity_check(self):
        if self._slot is not None and self._slot.kv.holds_kv:
            return
        self.inner.sanity_check()

    def _clear_pin(self) -> None:
        self._slot = None
        self._last_ids = None
        self._last_extra_key = None
        self._last_cache_salt = None

    def _drop_slot(self, reason: str) -> None:
        slot = self._slot
        self._clear_pin()
        if slot is None or not slot.kv.holds_kv:
            return
        self.free_kv_row(slot.kv, [(0, slot.kv.kv_allocated_len)])
        self.req_to_token_pool.free(slot)
        logger.info("sticky last-seq drop (%s)", reason)

    def _free_tail(self, kv: ReqKvInfo, prefix_len: int) -> None:
        self._free_kv_aligned(kv, prefix_len, kv.kv_allocated_len)
        kv.kv_allocated_len = prefix_len
        kv.kv_committed_len = min(kv.kv_committed_len, prefix_len)
        kv.swa_evicted_seqlen = min(kv.swa_evicted_seqlen, prefix_len)

    def _trim_overshoot(self, req: Req, finished_len: int) -> None:
        target = len(req.origin_input_ids) + finished_len
        if self.page_size > 1 and req.kv.swa_evicted_seqlen > target:
            target = (target // self.page_size) * self.page_size
        self._free_kv_aligned(req.kv, target, req.kv.kv_allocated_len)
        req.kv.kv_allocated_len = min(req.kv.kv_allocated_len, target)
        req.kv.kv_committed_len = min(req.kv.kv_committed_len, target)
        req.kv.swa_evicted_seqlen = min(req.kv.swa_evicted_seqlen, target)
        req.output_ids = req.output_ids[:finished_len]

    def _free_kv_aligned(self, kv: ReqKvInfo, target: int, end: int) -> None:
        if end <= target:
            return
        start = target
        if self.page_size > 1:
            start = ceil_align(start, self.page_size)
        self.free_kv_row(kv, [(start, end)])

    def __getattr__(self, name):
        return getattr(self.inner, name)
