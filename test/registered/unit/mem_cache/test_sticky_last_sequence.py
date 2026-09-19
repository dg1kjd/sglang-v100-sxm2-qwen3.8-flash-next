"""CPU tests for one-slot exact token-id continuation on ChunkCache."""

from __future__ import annotations

from array import array

import torch

from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.sticky_last_sequence import StickyLastSequenceCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _FakeAllocator(BaseTokenToKVPoolAllocator):
    def __init__(self, page_size: int = 1):
        super().__init__(
            size=1024,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.freed = []

    def clear(self):
        self.freed = []

    def alloc(self, need_size: int):
        raise NotImplementedError

    def free(self, free_index: torch.Tensor):
        self.freed.append(free_index.clone())


class _FakeReqToTokenPool:
    def __init__(self, req_to_token):
        self.req_to_token = req_to_token
        self.free_slots = []

    def free(self, req):
        self.free_slots.append(req.kv.req_pool_idx)
        req.kv.req_pool_idx = None


class _FakeInnerCache:
    def __init__(self, req_to_token_pool, allocator, page_size=1):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = allocator
        self.page_size = page_size
        self.disable = True
        self.finished = []
        self.matches = 0
        self.reset_calls = 0

    def is_chunk_cache(self):
        return True

    def match_prefix(self, params):
        self.matches += 1
        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64),
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
        )

    def cache_finished_req(self, req, is_insert=True, **kwargs):
        self.finished.append(req)

    def cache_unfinished_req(self, req, **kwargs):
        return None

    def reset(self):
        self.reset_calls += 1

    def evict(self, params):
        return None

    def evict_for_alloc(self, params):
        return None

    def inc_lock_ref(self, node):
        raise AssertionError("inner lock should not run for sticky sentinel")

    def dec_lock_ref(self, node, params=None):
        raise AssertionError("inner unlock should not run for sticky sentinel")

    def protected_size(self):
        return 0

    def evictable_size(self):
        return 0

    def pretty_print(self):
        return ""

    def supports_swa(self):
        return False

    def supports_mamba(self):
        return False

    def sanity_check(self):
        return None


class _FakeReq:
    def __init__(self, req_pool_idx, committed, allocated, origin, output=None):
        self.kv = ReqKvInfo(
            req_pool_idx=req_pool_idx,
            kv_committed_len=committed,
            kv_allocated_len=allocated,
            swa_evicted_seqlen=0,
            cache_protected_len=0,
        )
        self.origin_input_ids = list(origin)
        self.output_ids = list(output or [])
        self.extra_key = None
        self.cache_salt = None
        self.finished_reason = None
        self.finished_len = None
        self.to_finish = None

    @property
    def output_ids_through_stop(self):
        if self.finished_len is not None:
            return self.output_ids[: self.finished_len]
        return self.output_ids

    def detach_kv(self):
        kv, self.kv = self.kv, ReqKvInfo()
        return kv


def _key(ids, extra_key=None, cache_salt=None, limit=None):
    return RadixKey(
        token_ids=array("q", ids),
        extra_key=extra_key,
        cache_salt=cache_salt,
        limit=limit,
    )


class TestStickyLastSequence(CustomTestCase):
    def _make(self, page_size=1, row_len=128):
        req_to_token = torch.arange(row_len, dtype=torch.int32).reshape(1, row_len)
        pool = _FakeReqToTokenPool(req_to_token)
        allocator = _FakeAllocator(page_size=page_size)
        inner = _FakeInnerCache(pool, allocator, page_size=page_size)
        return StickyLastSequenceCache(inner), inner, allocator, pool

    def _pin(self, cache, origin, output, allocated=None):
        ids = list(origin) + list(output)
        n = len(ids)
        req = _FakeReq(
            req_pool_idx=0,
            committed=n,
            allocated=allocated if allocated is not None else n,
            origin=origin,
            output=output,
        )
        cache.cache_finished_req(req, kv_len_to_handle=n)
        return req, ids

    def test_exact_continuation_hits(self):
        cache, inner, allocator, pool = self._make()
        _, last = self._pin(cache, origin=list(range(8)), output=[8, 9])
        self.assertEqual(cache.session_held_tokens(), 10)
        self.assertEqual(cache.session_held_req_count(), 1)
        self.assertFalse(allocator.freed)
        self.assertEqual(pool.free_slots, [])

        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        result = cache.match_prefix(
            MatchPrefixParams(key=_key(last + [50, 51]), req=nxt)
        )
        self.assertEqual(inner.matches, 0)
        self.assertEqual(len(result.device_indices), 10)
        self.assertEqual(result.device_indices.tolist(), list(range(10)))
        self.assertEqual(nxt.kv.req_pool_idx, 0)
        self.assertEqual(result.cache_protected_len, 0)

    def test_shorter_prompt_misses_and_drops(self):
        cache, inner, allocator, pool = self._make()
        self._pin(cache, origin=list(range(8)), output=[8, 9])
        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        result = cache.match_prefix(
            MatchPrefixParams(key=_key(list(range(8))), req=nxt)
        )
        self.assertEqual(len(result.device_indices), 0)
        self.assertEqual(inner.matches, 1)
        self.assertIsNone(cache._last_ids)
        self.assertEqual(pool.free_slots, [0])
        self.assertTrue(allocator.freed)

    def test_partial_prefix_misses(self):
        cache, inner, _, _ = self._make()
        self._pin(cache, origin=list(range(8)), output=[8, 9])
        # Same first 6 tokens, then diverges — not an exact continuation.
        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        result = cache.match_prefix(
            MatchPrefixParams(key=_key(list(range(6)) + [99, 100]), req=nxt)
        )
        self.assertEqual(len(result.device_indices), 0)
        self.assertEqual(inner.matches, 1)
        self.assertIsNone(cache._last_ids)

    def test_different_tokens_miss(self):
        cache, inner, _, _ = self._make()
        self._pin(cache, origin=list(range(8)), output=[8, 9])
        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        result = cache.match_prefix(
            MatchPrefixParams(key=_key([7] + list(range(1, 12))), req=nxt)
        )
        self.assertEqual(len(result.device_indices), 0)
        self.assertEqual(inner.matches, 1)

    def test_miss_forgets_old_last_ids(self):
        cache, _, _, _ = self._make()
        self._pin(cache, origin=list(range(8)), output=[8, 9])
        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        cache.match_prefix(MatchPrefixParams(key=_key([1, 2, 3]), req=nxt))
        # A later request that *would* have continued the dropped sequence
        # must not resurrect the pin (pos-0 prefill already overwrote CSA2).
        later = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        result = cache.match_prefix(
            MatchPrefixParams(key=_key(list(range(8)) + [8, 9, 50]), req=later)
        )
        self.assertEqual(len(result.device_indices), 0)
        self.assertIsNone(later.kv.req_pool_idx)

    def test_extra_key_mismatch_misses(self):
        cache, inner, _, _ = self._make()
        _, last = self._pin(cache, origin=list(range(4)), output=[4])
        cache._last_extra_key = "lora-a"
        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        result = cache.match_prefix(
            MatchPrefixParams(key=_key(last + [9], extra_key="lora-b"), req=nxt)
        )
        self.assertEqual(len(result.device_indices), 0)
        self.assertEqual(inner.matches, 1)

    def test_abort_does_not_save(self):
        cache, inner, _, _ = self._make()
        req = _FakeReq(
            req_pool_idx=0,
            committed=4,
            allocated=4,
            origin=list(range(4)),
        )
        req.finished_reason = FINISH_ABORT("too long")
        cache.cache_finished_req(req, kv_len_to_handle=4)
        self.assertIsNone(cache._last_ids)
        self.assertEqual(inner.finished, [req])

    def test_reset_frees_pin(self):
        cache, inner, allocator, pool = self._make()
        self._pin(cache, origin=list(range(4)), output=[4, 5])
        cache.reset()
        self.assertIsNone(cache._last_ids)
        self.assertEqual(pool.free_slots, [0])
        self.assertTrue(allocator.freed)
        self.assertEqual(inner.reset_calls, 1)
        self.assertEqual(cache.session_held_tokens(), 0)

    def test_second_finish_updates_last_ids(self):
        cache, inner, _, _ = self._make()
        _, last = self._pin(cache, origin=list(range(6)), output=[6])
        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        cache.match_prefix(MatchPrefixParams(key=_key(last + [20, 21]), req=nxt))
        nxt.origin_input_ids = last + [20]
        nxt.output_ids = [21]
        nxt.kv.kv_committed_len = 9
        nxt.kv.kv_allocated_len = 9
        cache.cache_finished_req(nxt, kv_len_to_handle=9)
        self.assertEqual(cache._last_ids, tuple(last + [20, 21]))
        self.assertEqual(inner.finished, [])
        self.assertFalse(nxt.kv.holds_kv)

    def test_held_tokens_excluded_when_in_batch(self):
        cache, _, _, _ = self._make()
        self._pin(cache, origin=list(range(5)), output=[5])
        self.assertEqual(cache.session_held_tokens(), 6)
        self.assertEqual(cache.session_held_tokens(active_pool_idxs={0}), 0)
        self.assertEqual(cache.session_held_req_count(active_pool_idxs={0}), 0)
