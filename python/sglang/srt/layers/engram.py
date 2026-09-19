"""Engram: gated n-gram hash memory added to the hc residual stream.

Predecessors come from the live batch and per-request history of the n - 1
previous tokens. The token map, hash multipliers and prime layout must match
sglang.kernels.ops.embeddings.engram_hash to produce identical hash ids.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import mmap
import os
import re
import time
from typing import Optional

import msgspec
import numpy as np
import torch
from torch import nn

from sglang.kernels.ops.embeddings.engram_gate import fused_engram_gate
from sglang.kernels.ops.embeddings.engram_gather import engram_gather
from sglang.kernels.ops.embeddings.engram_hash import (
    MODE_DECODE,
    MODE_EXTEND,
    MODE_VERIFY,
    engram_commit_history,
    engram_hash_ids,
    engram_hash_ids_and_commit,
)
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.torch_quant import FP8_BLOCK_SIZE
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    dp_gather_replicate,
    dp_reduce_scatter_tensor,
    dp_scatter,
    get_attention_dp_size,
    get_global_dp_buffer_len,
    is_dp_gatherv_active,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import (
    get_model,
    get_parallel,
    get_platform,
    get_serving,
)
from sglang.srt.mem_cache.dsv41_host_placement import (
    DEFAULT_GPU_NUMA_NODE,
    GIB,
    EngramNumaError,
    bind_buffer_to_numa_node,
    memfd_create,
    mmap_hugetlb,
    read_huge_pages,
    read_numa_nodes,
    smaps_huge_kb,
)
from sglang.srt.utils import add_prefix
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)
_ENGRAM_ZERO_LOGGED = False

# cudaHostRegisterMapped: GPU kernels can UVA-load the host pointer. Flag 0
# only pins for memcpy and is not enough for SM70 discrete-GPU gathers.
_CUDA_HOST_REGISTER_MAPPED = 0x02


def engram_lookup_dtype(device: torch.device | str | int | None = None) -> torch.dtype:
    """Dequant/gate activation dtype: fp16 on SM70 (no BF16), bf16 on Hopper+."""
    if device is not None:
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type == "cuda" and torch.cuda.is_available():
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
            return (
                torch.float16
                if torch.cuda.get_device_capability(idx)[0] == 7
                else torch.bfloat16
            )
    try:
        platform = get_platform()
        if platform.is_sm70:
            return torch.float16
    except Exception:
        pass
    return torch.bfloat16


def _find_next_prime(start: int, seen_primes: set[int]) -> int:
    from sympy import isprime

    candidate = start + 1
    while not isprime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Token id -> compressed id, plus the compressed vocab size. The size feeds every
    hash multiplier, so a mismatch rehashes the whole table."""
    from tokenizers import Regex, normalizers

    # A private-use sentinel keeps a lone-space token from collapsing to "" under Strip().
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # A partial UTF-8 byte token has nothing to normalize; key it by its raw form.
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, vocab_size: int
) -> torch.Tensor:
    """One odd multiplier per (layer, lookback) from a per-layer RNG, bounded so that
    token_id * multiplier fits in int64."""
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


class EngramLayout(msgspec.Struct, frozen=True):
    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [layer][n-gram size][head]
    n_heads: int
    head_dim: int

    @classmethod
    def build(
        cls,
        layer_ids: tuple[int, ...],
        num_embeddings: tuple[int, ...],
        max_ngram_size: int,
        n_heads: int,
        head_dim: int,
        vocab_size: int,
    ) -> EngramLayout:
        """Primes are drawn in (layer, n-gram size, head) order from one shared
        ascending sequence starting above vocab_size - 1."""
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], vocab_size - 1
                for _ in range(n_heads):
                    current = _find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=num_embeddings,
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=head_dim,
        )


def build_engram_layout(config) -> Optional[EngramLayout]:
    layer_ids = tuple(config.engram_layer_ids)
    if not layer_ids:
        return None
    return EngramLayout.build(
        layer_ids=layer_ids,
        num_embeddings=tuple(config.engram_num_embeddings),
        max_ngram_size=config.engram_max_ngram_size,
        n_heads=config.engram_n_heads,
        head_dim=config.engram_head_dim,
        vocab_size=config.engram_vocab_size,
    )


def compute_engram_hash_ids(
    tokens: torch.Tensor,
    blocked: torch.Tensor,
    pad_id: int,
    token_map: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """tokens [T, n]: column 0 is the token itself, column s its s-th predecessor;
    blocked [T, n] marks look-back that ran off the sequence start. Returns
    [T, n_engram_layers, (n - 1) * n_heads] row ids into each layer's table."""
    compressed = torch.where(blocked, pad_id, token_map[tokens])
    products = compressed.unsqueeze(1) * multipliers
    # XOR the multiplied ids one look-back at a time: after step i the running value
    # is the (i + 1)-gram hash, bucketed by that n-gram size's primes.
    rolling, hashes = products[..., 0], []
    for i in range(1, tokens.shape[-1]):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + offsets


class EngramHasher(nn.Module):
    """Hash ids for every token of a forward batch, [T, n_engram_layers, n_hash_cols]."""

    def __init__(
        self,
        layout: EngramLayout,
        tokenizer,
        pad_id: int,
        compressed_vocab_size: int,
    ):
        super().__init__()
        self.max_ngram_size = layout.max_ngram_size
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        assert vocab_size == compressed_vocab_size, (
            f"the tokenizer normalizes to {vocab_size} distinct tokens but the config "
            f"expects {compressed_vocab_size}; every hash multiplier depends on it"
        )
        self.pad_id = token_map[pad_id]
        flat = [
            [p for per_ngram in layer for p in per_ngram] for layer in layout.primes
        ]
        offsets = np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat])
        multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, vocab_size
        )
        self.register_buffer("token_map", torch.tensor(token_map), persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)
        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(offsets), persistent=False)
        self.image_token_id: Optional[int] = None
        self.history: Optional[torch.Tensor] = None
        self.pad_row = 0

    def init_history(self, num_req_slots: int, device) -> None:
        """Allocate oldest-first history with a spare row for graph padding.

        Extend reads scheduler-provided predecessors after prefix hits or
        retraction. Decode commits in forward; verify commits after acceptance.
        """
        self.history = torch.zeros(
            num_req_slots + 1,
            self.max_ngram_size - 1,
            dtype=torch.int32,
            device=device,
        )
        self.pad_row = num_req_slots

    @classmethod
    def from_config(cls, config, layout: EngramLayout) -> EngramHasher:
        # The compressed token map is built with the HF normalizers, so the HF
        # tokenizer backend is used here whatever the serving backend is.
        tokenizer = get_tokenizer(
            get_serving().tokenizer_path,
            tokenizer_mode=get_serving().tokenizer_mode,
            trust_remote_code=get_model().trust_remote_code,
            revision=get_model().revision,
            tokenizer_backend="huggingface",
        )
        result = cls(
            layout,
            tokenizer,
            config.engram_pad_token_id,
            config.engram_compressed_vocab_size,
        )
        result.image_token_id = (
            config.image_token_id
            if config.model_type == "deepseek_v41" and config.vision_n_layers > 0
            else None
        )
        return result

    def forward(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        assert self.history is not None, "EngramHasher.init_history was not called"
        n = self.max_ngram_size
        num_tokens = input_ids.shape[0]
        if num_tokens == 0:
            return torch.empty(
                (0, self.primes.shape[0], self.offsets.shape[1]),
                dtype=torch.int64,
                device=input_ids.device,
            )
        mode = forward_batch.forward_mode
        req_slots = forward_batch.req_pool_indices
        bs = req_slots.shape[0]
        device = input_ids.device
        # Which request each token belongs to and where its run starts; tokens at or
        # past num_real are graph padding. History rows come from self.history via
        # req_slots unless the scheduler supplied this extend's predecessors.
        num_real, block, row, starts = num_tokens, 1, None, None
        history, hist_via_slots = self.history, True
        if mode.is_decode():
            kmode = MODE_DECODE
            commit_rows, commit_last = req_slots, None
        elif mode.is_target_verify():
            block = int(forward_batch.spec_info.draft_token_num)
            assert num_tokens == bs * block, (
                "engram target-verify expects one equal block per request, got "
                f"{num_tokens} tokens for {bs} requests of {block}"
            )
            kmode = MODE_VERIFY
            commit_rows = commit_last = None
        else:
            assert mode.is_extend(), (
                f"engram serves extend, target-verify and decode, not {mode}"
            )
            lens = forward_batch.extend_seq_lens.to(torch.int64)
            starts = forward_batch.extend_start_loc.to(torch.int64)
            row = torch.repeat_interleave(torch.arange(bs, device=device), lens)
            num_real = row.shape[0]
            kmode = MODE_EXTEND
            if forward_batch.ngram_history is not None:
                history, hist_via_slots = forward_batch.ngram_history, False
            commit_rows = torch.where(lens > 0, req_slots, self.pad_row)
            commit_last = (starts + lens - 1).clamp(0, num_tokens - 1)

        if input_ids.is_cuda and torch.version.cuda is not None:
            if kmode == MODE_DECODE:
                # out_cache_loc 0 marks the CUDA-graph padded rows that must not commit.
                assert forward_batch.out_cache_loc is not None
                return engram_hash_ids_and_commit(
                    input_ids,
                    forward_batch.positions,
                    history=self.history,
                    req_slots=req_slots,
                    out_cache_loc=forward_batch.out_cache_loc,
                    token_map=self.token_map,
                    multipliers=self.multipliers,
                    primes=self.primes,
                    offsets=self.offsets,
                    pad_id=self.pad_id,
                    image_token_id=self.image_token_id,
                    mm_pad_shift=MM_PAD_SHIFT_VALUE,
                )
            hash_ids, tokens = engram_hash_ids(
                input_ids,
                forward_batch.positions,
                mode=kmode,
                history=history,
                token_map=self.token_map,
                multipliers=self.multipliers,
                primes=self.primes,
                offsets=self.offsets,
                pad_id=self.pad_id,
                num_real=num_real,
                req_slots=req_slots if hist_via_slots else None,
                block=block,
                row=row,
                starts=starts,
                image_token_id=self.image_token_id,
                mm_pad_shift=MM_PAD_SHIFT_VALUE,
            )
        else:
            hash_ids, tokens = self._torch_hash_ids(
                input_ids,
                forward_batch.positions,
                kmode,
                history[req_slots] if hist_via_slots else history,
                num_real,
                block,
                row,
                starts,
            )
        if commit_rows is not None:
            # Padded rows must not overwrite a live request's history.
            last_tokens = tokens if commit_last is None else tokens[commit_last]
            out_loc = forward_batch.out_cache_loc
            if out_loc is not None:
                if commit_last is not None:
                    out_loc = out_loc[commit_last]
                commit_rows = torch.where(out_loc == 0, self.pad_row, commit_rows)
            self.history[commit_rows] = (
                last_tokens[:, : n - 1].flip(-1).to(self.history.dtype)
            )
        return hash_ids

    def _torch_hash_ids(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kmode: int,
        history: torch.Tensor,
        num_real: int,
        block: int,
        row: Optional[torch.Tensor],
        starts: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Non-CUDA fallback of the hash kernel: predecessor table [T, n] and hash ids.
        ``history`` is already the per-request [bs, n - 1] rows of this batch."""
        n = self.max_ngram_size
        num_tokens = input_ids.shape[0]
        device = input_ids.device
        input_ids = input_ids.to(torch.int64)
        shifts = torch.arange(n, device=device)
        if kmode == MODE_DECODE:
            tokens = torch.cat(
                [input_ids.unsqueeze(-1), history.to(torch.int64).flip(-1)], dim=-1
            )
        else:
            t = torch.arange(num_real, device=device)
            if kmode == MODE_VERIFY:
                row = t // block
                offset = t - row * block
            else:
                offset = t - starts[row]
            # Predecessor at shift s of token t: an earlier token of the same run
            # when s <= offset, else history[row, -(s - offset)].
            from_batch = shifts.unsqueeze(0) <= offset.unsqueeze(-1)
            in_batch = input_ids[(t.unsqueeze(-1) - shifts).clamp_min(0)]
            hist_col = (n - 2 - (shifts.unsqueeze(0) - offset.unsqueeze(-1) - 1)).clamp(
                0, n - 2
            )
            from_hist = history.to(torch.int64)[row].gather(1, hist_col)
            tokens = torch.where(from_batch, in_batch, from_hist)
        if num_real < num_tokens:
            tokens = torch.cat([tokens, tokens.new_zeros(num_tokens - num_real, n)])
        positions = positions.to(torch.int64)
        blocked = positions.unsqueeze(-1) < shifts
        if num_real < num_tokens:
            blocked[num_real:] = True
        if self.image_token_id is not None:
            # Scheduler-provided history still carries the multimodal pad ids.
            tokens = tokens.masked_fill(
                tokens >= MM_PAD_SHIFT_VALUE, self.image_token_id
            )
            # Once a lookback hits an image, every older predecessor is PAD.
            blocked = (
                (blocked | (tokens == self.image_token_id))
                .to(torch.int32)
                .cummax(-1)
                .values.bool()
            )
        hash_ids = compute_engram_hash_ids(
            tokens,
            blocked,
            self.pad_id,
            self.token_map,
            self.multipliers,
            self.primes,
            self.offsets,
        )
        return hash_ids, tokens

    def commit_after_verify(
        self,
        verify_ids_2d: torch.Tensor,
        req_pool_indices: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Commit anchor + accepted drafts; the bonus is the next block's anchor."""
        assert self.history is not None, "EngramHasher.init_history was not called"
        if self.history.is_cuda:
            engram_commit_history(
                self.history, verify_ids_2d, req_pool_indices, commit_lens
            )
            return
        n1 = self.max_ngram_size - 1
        req = req_pool_indices.to(torch.int64)
        window = torch.cat(
            [self.history[req], verify_ids_2d.to(self.history.dtype)], dim=1
        )
        cols = commit_lens.to(torch.int64).unsqueeze(-1) + torch.arange(
            n1, device=window.device
        )
        self.history[req] = window.gather(1, cols)


_THP_DIR = "/sys/kernel/mm/transparent_hugepage"


def _thp_mode(knob: str) -> str:
    """Active mode of a transparent_hugepage sysfs knob ("" if unreadable)."""
    try:
        with open(f"{_THP_DIR}/{knob}") as f:
            m = re.search(r"\[(\w+)\]", f.read())
        return m.group(1) if m else ""
    except OSError:
        return ""


def _huge_pages_backing(addr: int) -> tuple[int, int]:
    """(mapped_kB, huge_kB) of the VMA holding addr, from /proc/self/smaps.

    Counts both THP (2 MiB) and hugetlb (1 GiB ``Private_Hugetlb`` /
    ``KernelPageSize``).
    """
    rss, thp, hugetlb = smaps_huge_kb(addr)
    return rss, thp + hugetlb


_page_cache_dropped = False


def drop_checkpoint_page_cache(model_path: Optional[str] = None) -> tuple[int, int]:
    """Drop checkpoint page cache with posix_fadvise(DONTNEED); return (files, bytes).

    Cached checkpoint pages can fragment host memory and fail hugepage
    faults for Engram tables. Optional; gated by
    ``SGLANG_ENABLE_DSV41_ENGRAM_DROP_PAGE_CACHE``.
    """
    if model_path is None:
        try:
            model_path = get_model().model_path
        except (ValueError, AttributeError):
            # No published runtime context (unit tests, offline tools): nothing to drop.
            return 0, 0
    files, nbytes = 0, 0
    for f in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        try:
            fd = os.open(f, os.O_RDONLY)
        except OSError:
            continue
        try:
            nbytes += os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            files += 1
        finally:
            os.close(fd)
    return files, nbytes


def _drop_page_cache_once(reason: str) -> None:
    global _page_cache_dropped
    if _page_cache_dropped or not envs.SGLANG_ENABLE_DSV41_ENGRAM_DROP_PAGE_CACHE.get():
        return
    _page_cache_dropped = True
    files, nbytes = drop_checkpoint_page_cache()
    logger.info(
        "engram host table: dropped the page cache of %d checkpoint files (%.0f GiB) %s",
        files,
        nbytes / 2**30,
        reason,
    )


# Cumulative Engram host bytes already bound, so the second ~94 GiB table
# cannot silently overflow node 1 onto node 0. Shared memfd copies are
# reserved once (rank 0); private shards reserve per rank.
_engram_numa_used: dict[int, int] = {}
_engram_shared_node: dict[str, int] = {}


def _choose_engram_numa_node(
    charge_bytes: int,
    *,
    layout: str,
    rank_in_group: int,
    name: str,
) -> Optional[int]:
    """Bind this mapping to the GPU NUMA node, or split, or fail loud.

    ``charge_bytes`` is the *cluster* footprint of this table: the full table
    for a shared memfd, or ``shard * tp_size`` for a private shard so every
    rank fails together instead of each thinking its 1/8th fits.

    Returns None when NUMA binding is disabled (env node < 0).
    """
    preferred = int(envs.SGLANG_DSV41_ENGRAM_NUMA_NODE.get())
    if preferred < 0:
        return None
    if layout == "shared" and rank_in_group != 0:
        node = _engram_shared_node.get(name)
        if node is None:
            raise EngramNumaError(
                f"engram shared table {name}: rank 0 has not reserved a NUMA node"
            )
        return node
    allow_split = bool(envs.SGLANG_DSV41_ENGRAM_NUMA_SPLIT.get())
    nodes = read_numa_nodes(documented_fallback=True)
    if preferred not in nodes:
        raise EngramNumaError(
            f"SGLANG_DSV41_ENGRAM_NUMA_NODE={preferred} is not a visible NUMA "
            f"node (have {sorted(nodes)}). Set it to a node that exists on "
            f"this host."
        )
    used = _engram_numa_used.get(preferred, 0)
    cap = nodes[preferred].total_bytes
    if used + charge_bytes <= cap:
        _engram_numa_used[preferred] = used + charge_bytes
        if layout == "shared":
            _engram_shared_node[name] = preferred
        return preferred
    if not allow_split:
        raise EngramNumaError(
            f"Engram host mapping of {charge_bytes / 2**30:.1f} GiB does not fit "
            f"NUMA node {preferred} "
            f"({nodes[preferred].total_gib:.1f} GiB total, {used / 2**30:.1f} GiB "
            f"already placed). Set SGLANG_DSV41_ENGRAM_NUMA_SPLIT=1 to place "
            f"overflow tables on another node (cross-socket tax), or put more "
            f"DRAM on node {preferred} — Linux must not silently migrate pages."
        )
    others = [n for n in sorted(nodes) if n != preferred]
    if not others:
        raise EngramNumaError("NUMA split requested but no other node is visible")
    alt = others[0]
    alt_used = _engram_numa_used.get(alt, 0)
    if alt_used + charge_bytes > nodes[alt].total_bytes:
        raise EngramNumaError(
            f"Engram overflow mapping of {charge_bytes / 2**30:.1f} GiB fits neither "
            f"NUMA node {preferred} nor {alt}"
        )
    _engram_numa_used[alt] = alt_used + charge_bytes
    if layout == "shared":
        _engram_shared_node[name] = alt
    logger.warning(
        "engram host table: placing %.1f GiB on NUMA node %d instead of GPU "
        "node %d (SGLANG_DSV41_ENGRAM_NUMA_SPLIT=1); gathers are cross-socket",
        charge_bytes / 2**30,
        alt,
        preferred,
    )
    return alt


class _HostTable:
    """Host-memory backing for one engram table.

    Layouts:
      shared   one memfd holding every row, mapped by all ranks of the group;
               rank 0 creates it, the others open it through /proc/<pid>/fd.
               No all-reduce. 1 GiB hugetlb memfd when the table is >= 1 GiB.
      private  one anonymous mapping per rank holding only its own rows.
               Tiny mappings stay on 4K/THP so unit tests do not grab a 1G page.

    Placement is NUMA-bound to SGLANG_DSV41_ENGRAM_NUMA_NODE.
    """

    def __init__(
        self, layout: str, nbytes: int, name: str, group, pin: bool, tp_size: int = 1
    ):
        assert layout in ("shared", "private"), layout
        self.layout = layout
        self.nbytes = nbytes
        self.group = group
        self.dirty = False
        self.registered = False
        self.numa_node: Optional[int] = None
        self.hugetlb = False
        self.map_bytes = nbytes
        rank_in_group = getattr(group, "rank_in_group", 0)
        charge = nbytes if layout == "shared" else nbytes * max(int(tp_size), 1)
        use_hugetlb = bool(envs.SGLANG_ENABLE_DSV41_ENGRAM_HUGETLB.get()) and nbytes >= GIB
        page_bytes = GIB
        if use_hugetlb:
            _total, size_kb = read_huge_pages()
            if size_kb == 2048:
                page_bytes = 2 * 1024 * 1024
            elif size_kb != 1048576:
                raise EngramNumaError(
                    f"Hugepagesize is {size_kb} kB; Engram expects 1G (or 2M) hugetlb"
                )

        if layout == "shared":
            self.fd = None
            if use_hugetlb:
                meta = None
                if rank_in_group == 0:
                    node = _choose_engram_numa_node(
                        charge, layout=layout, rank_in_group=0, name=name
                    )
                    if node is None:
                        node = DEFAULT_GPU_NUMA_NODE
                    mm, fd, map_bytes = mmap_hugetlb(
                        nbytes, node=node, page_bytes=page_bytes, shared=True, name=name
                    )
                    self.mm, self.fd, self.map_bytes = mm, fd, map_bytes
                    self.numa_node = node
                    self.hugetlb = True
                    meta = (os.getpid(), fd, node, map_bytes)
                pid, owner_fd, node, map_bytes = group.broadcast_object(meta, src=0)
                group.barrier()
                if rank_in_group != 0:
                    fd = os.open(f"/proc/{pid}/fd/{owner_fd}", os.O_RDWR)
                    mm, _, _ = mmap_hugetlb(
                        nbytes,
                        node=node,
                        page_bytes=page_bytes,
                        shared=True,
                        fd=fd,
                        name=name,
                    )
                    self.mm, self.fd, self.map_bytes = mm, fd, map_bytes
                    self.numa_node = node
                    self.hugetlb = True
            else:
                self.fd = self._open_shared_fd(nbytes, name)
                self.mm = mmap.mmap(
                    self.fd,
                    nbytes,
                    flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                )
        else:
            self.fd = None
            if use_hugetlb:
                node = _choose_engram_numa_node(
                    charge, layout=layout, rank_in_group=rank_in_group, name=name
                )
                if node is None:
                    node = DEFAULT_GPU_NUMA_NODE
                mm, _, map_bytes = mmap_hugetlb(
                    nbytes, node=node, page_bytes=page_bytes, shared=False, name=name
                )
                self.mm, self.map_bytes = mm, map_bytes
                self.numa_node = node
                self.hugetlb = True
            else:
                self.mm = mmap.mmap(
                    -1,
                    nbytes,
                    flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                )
        if not self.hugetlb:
            # Advisory before the first touch: THP pages at fault time.
            self.mm.madvise(mmap.MADV_HUGEPAGE)
        self.bytes = torch.frombuffer(self.mm, dtype=torch.uint8)
        if not self.hugetlb:
            if layout == "private":
                self.numa_node = _choose_engram_numa_node(
                    charge, layout=layout, rank_in_group=rank_in_group, name=name
                )
                if self.numa_node is not None:
                    bind_buffer_to_numa_node(
                        self.bytes.data_ptr(), nbytes, self.numa_node
                    )
                _drop_page_cache_once("before pre-faulting the private shard")
                np.frombuffer(self.mm, dtype=np.uint8)[:: mmap.PAGESIZE] = 0
            else:
                node = None
                if rank_in_group == 0:
                    node = _choose_engram_numa_node(
                        charge, layout=layout, rank_in_group=0, name=name
                    )
                    if node is not None:
                        bind_buffer_to_numa_node(self.bytes.data_ptr(), nbytes, node)
                node = group.broadcast_object(node, src=0)
                group.barrier()
                if rank_in_group != 0:
                    if node is not None:
                        bind_buffer_to_numa_node(self.bytes.data_ptr(), nbytes, node)
                self.numa_node = node
        if pin:
            err = torch.cuda.cudart().cudaHostRegister(
                self.bytes.data_ptr(), nbytes, _CUDA_HOST_REGISTER_MAPPED
            )
            if int(err) != 0:
                raise RuntimeError(f"cudaHostRegister({nbytes} bytes) failed: {err}")
            self.registered = True

    @staticmethod
    def choose_layout(requested: str) -> str:
        if requested != "auto":
            return requested
        total, size_kb = read_huge_pages()
        if (
            envs.SGLANG_ENABLE_DSV41_ENGRAM_HUGETLB.get()
            and total > 0
            and size_kb >= 2048
        ):
            return "shared"
        if _thp_mode("shmem_enabled") in ("advise", "always", "within_size", "force"):
            return "shared"
        if _thp_mode("enabled") in ("madvise", "always"):
            return "private"
        return "shared"

    def _open_shared_fd(self, nbytes: int, name: str) -> int:
        owner = None
        if self.group.rank_in_group == 0:
            fd = memfd_create(name, 0)
            os.ftruncate(fd, nbytes)
            owner = (os.getpid(), fd)
        pid, owner_fd = self.group.broadcast_object(owner, src=0)
        if self.group.rank_in_group == 0:
            return fd
        try:
            return os.open(f"/proc/{pid}/fd/{owner_fd}", os.O_RDWR)
        except OSError as e:
            raise RuntimeError(
                "engram host table: cannot open rank 0's memfd through /proc; the "
                "TP ranks must share a PID namespace"
            ) from e

    def _collapse(self, tries: int = 3) -> None:
        """madvise(MADV_COLLAPSE): synchronously fold whatever is still on base pages
        into huge pages. Anonymous memory only; shmem obeys shmem_enabled and
        refuses. EAGAIN means compaction ran out of time, so it is retried a few
        times; anything else is logged and left."""
        MADV_COLLAPSE = 25  # Linux >= 6.1; not in Python's mmap module
        libc = ctypes.CDLL(None, use_errno=True)
        libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        for attempt in range(tries):
            rc = libc.madvise(
                ctypes.c_void_p(self.bytes.data_ptr()),
                ctypes.c_size_t(self.nbytes),
                MADV_COLLAPSE,
            )
            if rc == 0:
                return
            err = ctypes.get_errno()
            if (
                err != 11 or attempt == tries - 1
            ):  # EAGAIN is the only one worth retrying
                logger.info("engram host table: MADV_COLLAPSE errno %d", err)
                return
            time.sleep(1.0)

    def finish_load(self, label: str):
        if not self.dirty:
            return
        self.dirty = False
        if self.layout == "shared":
            self.group.barrier()
        mapped_kb, huge_kb = _huge_pages_backing(self.bytes.data_ptr())
        if (
            not self.hugetlb
            and self.layout == "private"
            and huge_kb < mapped_kb * 0.98
        ):
            # The loader's own reads refilled the page cache; empty it again so the
            # collapse can find contiguous memory.
            if envs.SGLANG_ENABLE_DSV41_ENGRAM_DROP_PAGE_CACHE.get():
                drop_checkpoint_page_cache()
            self._collapse()
            mapped_kb, huge_kb = _huge_pages_backing(self.bytes.data_ptr())
        pct = 100.0 * huge_kb / mapped_kb if mapped_kb else 0.0
        numa = f", numa_node={self.numa_node}" if self.numa_node is not None else ""
        msg = (
            f"engram host table {label}: layout={self.layout}{numa}, "
            f"{mapped_kb / 2**10:.0f} MiB resident, {huge_kb / 2**10:.0f} MiB in huge pages "
            f"({pct:.0f}%){', pinned' if self.registered else ', unpinned (ATS)'}"
        )
        if huge_kb == 0:
            knob = "shmem_enabled" if self.layout == "shared" else "enabled"
            logger.warning(
                "%s. No huge pages: expect ~10x slower lookups (one TLB miss per row); "
                "transparent_hugepage/%s is '%s'",
                msg,
                knob,
                _thp_mode(knob) or "unreadable",
            )
        else:
            logger.info(msg)


class EngramEmbedding(nn.Module):
    """One layer's fp8 hash table with e8m0 block scales, dequantized on lookup.

    Default: rows sharded over the TP group in device memory; each rank gathers
    the rows it owns, zeroes the rest and the all-reduce reassembles the lookup.
    With SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE the table lives in host memory and
    the GPU gathers rows over the CPU link -- either one shared copy with no
    all-reduce, or one private shard per rank (see _HostTable). Loading is
    sharded in every layout: a rank writes only its own row range.
    """

    def __init__(self, num_embeddings: int, dim: int, layer_id: int):
        super().__init__()
        self.dim = dim
        self.tp_size = get_parallel().tp_size
        tp_rank = get_parallel().tp_rank
        self.row_start = num_embeddings * tp_rank // self.tp_size
        row_end = num_embeddings * (tp_rank + 1) // self.tp_size
        self.rows = row_end - self.row_start
        self.host_table: Optional[_HostTable] = None
        if envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get():
            self._init_host_table(num_embeddings, dim, layer_id)
        else:
            self.weight = nn.Parameter(
                torch.empty(self.rows, dim, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.scale = nn.Parameter(
                torch.empty(
                    self.rows, dim // FP8_BLOCK_SIZE, dtype=torch.float8_e8m0fnu
                ),
                requires_grad=False,
            )
        self.weight.weight_loader = self._load_rows
        self.scale.weight_loader = self._load_rows

    def _init_host_table(self, num_embeddings: int, dim: int, layer_id: int):
        layout = _HostTable.choose_layout(
            envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.get()
        )
        n = num_embeddings if layout == "shared" else self.rows
        w_bytes = n * dim
        s_bytes = n * (dim // FP8_BLOCK_SIZE)
        self.host_table = _HostTable(
            layout,
            max(1, w_bytes + s_bytes),  # mmap requires storage even for an empty shard.
            f"sglang_engram_{layer_id}",
            get_tp_group(),
            pin=envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN.get(),
            tp_size=self.tp_size,
        )
        raw = self.host_table.bytes[: w_bytes + s_bytes]
        weight = raw[:w_bytes].view(torch.float8_e4m3fn).view(n, dim)
        scale = raw[w_bytes:].view(torch.float8_e8m0fnu).view(n, dim // FP8_BLOCK_SIZE)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)

    @property
    def _shared(self) -> bool:
        return self.host_table is not None and self.host_table.layout == "shared"

    def _load_rows(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        rows = slice(self.row_start, self.row_start + self.rows)
        if self._shared:
            param.data[rows].copy_(loaded_weight[rows])
        else:
            param.data.copy_(loaded_weight[rows])
        if self.host_table is not None:
            self.host_table.dirty = True

    def finish_load(self, label: str = ""):
        """Barrier (shared layout) once every rank has written its rows; log how
        the table ended up backed."""
        if self.host_table is not None:
            self.host_table.finish_load(label)

    def gather_rows(self, indices: torch.Tensor) -> torch.Tensor:
        """Gather this rank's rows. No TP all-reduce. Safe on a side stream."""
        if self._shared:
            if indices.shape[0] == 0:
                return self._empty(indices)
            out = self._empty(indices)
            engram_gather(
                self.weight.data_ptr(),
                self.scale.data_ptr(),
                indices.reshape(-1),
                out.view(-1, self.dim),
                self.dim,
                FP8_BLOCK_SIZE,
            )
            return out
        return self._owned_rows(indices)

    def forward(
        self,
        indices: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        *,
        cp_all_tokens: bool = False,
    ) -> torch.Tensor:
        if self._shared:
            return self.gather_rows(indices)
        if cp_all_tokens and self.tp_size > 1:
            # Prefill CP: gather the hash ids over the CP group first so every
            # TP rank looks up the same indices, then keep this rank's slice.
            parallel = get_parallel()
            local_rows = indices.shape[0]
            all_indices = indices.new_empty(
                (parallel.attn_cp_size * local_rows, *indices.shape[1:])
            )
            attn_cp_all_gather_into_tensor(all_indices, indices.contiguous())
            start = parallel.attn_cp_rank * local_rows
            return self._lookup(all_indices)[start : start + local_rows]
        if self.tp_size > 1 and get_attention_dp_size() > 1:
            return self._dp_sharded_lookup(indices, forward_batch)
        return self._lookup(indices)

    def _lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Lookup when every TP rank holds the same indices: the device and
        private host shards zero unowned rows and the all-reduce reassembles."""
        if indices.shape[0] == 0:
            return self._empty(indices)
        values = self._owned_rows(indices)
        if self.tp_size > 1:
            values = tensor_model_parallel_all_reduce(values)
        return values

    def _empty(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            *indices.shape,
            self.dim,
            dtype=engram_lookup_dtype(indices.device),
            device=indices.device,
        )

    def _owned_rows(self, indices: torch.Tensor) -> torch.Tensor:
        """Rows of `indices` this rank's shard holds, zero for the rest."""
        if self.rows == 0:
            return self._empty(indices).zero_()
        if self.host_table is None and (
            not indices.is_cuda or torch.version.cuda is None
        ):
            local = indices - self.row_start
            owned = (local >= 0) & (local < self.rows)
            local = local.masked_fill(~owned, 0)
            # Torch 2.9 cannot index float8 on CPU; dequant the table first.
            rows = self.weight.float()[local].unflatten(-1, (-1, FP8_BLOCK_SIZE))
            values = (rows * self.scale.float()[local].unsqueeze(-1)).flatten(-2)
            return values.to(engram_lookup_dtype(indices.device)).masked_fill(
                ~owned.unsqueeze(-1), 0
            )
        out = self._empty(indices)
        engram_gather(
            self.weight.data_ptr(),
            self.scale.data_ptr(),
            indices.reshape(-1),
            out.view(-1, self.dim),
            self.dim,
            FP8_BLOCK_SIZE,
            row_lo=self.row_start,
            row_hi=self.row_start + self.rows,
        )
        return out

    def _dp_sharded_lookup(
        self, indices: torch.Tensor, forward_batch: Optional[ForwardBatch]
    ) -> torch.Tensor:
        """Gather DP ranks' indices before looking up TP-sharded rows.

        Every rank joins, including idle ranks with zero rows; the reduced result
        is sliced back to each rank's local tokens.
        """
        assert forward_batch is not None, "the DP engram lookup needs the batch"
        rows = get_global_dp_buffer_len()
        ids_global = torch.empty(
            (rows, *indices.shape[1:]), dtype=indices.dtype, device=indices.device
        )
        # The MAX_LEN gather may zero its local input in place, hence the clone.
        dp_gather_replicate(ids_global, indices.clone(), forward_batch)
        if rows == 0:
            return self._empty(indices)
        values = self._owned_rows(ids_global).view(rows, -1)
        local = torch.empty(
            (indices.shape[0], values.shape[1]),
            dtype=values.dtype,
            device=values.device,
        )
        # MAX_LEN or gatherv-sized slices use reduce-scatter;
        # otherwise scatter the summed buffer to the DP ranks.
        padding = forward_batch.dp_padding_mode
        if (
            padding is not None
            and padding.is_max_len()
            and self.tp_size == get_attention_dp_size()
            and rows == self.tp_size * local.shape[0]
        ) or is_dp_gatherv_active():
            dp_reduce_scatter_tensor(local, values)
        else:
            dp_scatter(local, tensor_model_parallel_all_reduce(values), forward_batch)
        return local.view(*indices.shape, self.dim)


def engram_gate(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float,
) -> torch.Tensor:
    """x [T, hc_mult, dim]; kv [T, (hc_mult + 1) * dim] holds one key per hc copy
    followed by the shared value. Adds the gated value to every copy.

    The fused kernel serves every token count on CUDA; the torch path below is the
    non-CUDA fallback and materializes fp32 copies of x, key and value."""
    if (
        x.is_cuda
        and torch.version.cuda is not None
        and x.ndim == 3
        and kv.shape == (x.shape[0], (x.shape[1] + 1) * x.shape[2])
        and x.dtype == kv.dtype
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and q_weight.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and k_weight.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and all(t.is_contiguous() for t in (x, kv, q_weight, k_weight))
    ):
        return fused_engram_gate(x, kv, q_weight, k_weight, eps, clamp_value)
    hc_mult, dim = x.shape[-2:]
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    weight = q_weight.float() * k_weight.float()
    h = x.float()
    # Normalized per (token, hc copy) over dim, not jointly over the copies.
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    # Signed square root before the sigmoid, matching the training kernel.
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


class Engram(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        layout: EngramLayout,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        dim, hc_mult = config.hidden_size, config.hc_mult
        self.embed = EngramEmbedding(
            layout.num_embeddings[self.layer_hash_index], layout.head_dim, layer_id
        )
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = ReplicatedLinear(
            n_hash_cols * layout.head_dim,
            dim * (hc_mult + 1),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wkv", prefix),
        )
        self.q_weight = nn.Parameter(torch.ones(hc_mult, dim), requires_grad=False)
        self.k_weight = nn.Parameter(torch.ones(hc_mult, dim), requires_grad=False)

    def forward(
        self,
        x: torch.Tensor,
        hash_ids: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        *,
        cp_all_tokens: bool = False,
    ) -> torch.Tensor:
        """x [T, hc_mult, dim]; hash_ids [T, n_hash_cols] for this layer."""
        if envs.SGLANG_DSV41_ENGRAM_ZERO.get():
            global _ENGRAM_ZERO_LOGGED
            if not _ENGRAM_ZERO_LOGGED:
                logger.warning(
                    "SGLANG_DSV41_ENGRAM_ZERO: skipping Engram gather/gate "
                    "(WO-11 ablation; residual unchanged)"
                )
                _ENGRAM_ZERO_LOGGED = True
            return x
        # The lookup runs first even for an idle DP-attention batch: under DP
        # attention it is a collective every rank has to join.
        emb = self.embed(hash_ids, forward_batch, cp_all_tokens=cp_all_tokens)
        return self._wkv_gate(x, emb)

    def _wkv_gate(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            # Nothing to gate, and the MXFP8 quantize behind wkv rejects an
            # empty M.
            return x
        kv, _ = self.wkv(emb.flatten(-2))
        return self.apply_gate(x, kv)

    def finish_from_gathered(
        self, x: torch.Tensor, gathered: torch.Tensor
    ) -> torch.Tensor:
        """Consume a side-stream ``embed.gather_rows``: TP all-reduce + wkv + gate.

        Collectives stay on the caller's stream. Do not pass ``embed.forward``
        output here: that path already all-reduced.
        """
        if envs.SGLANG_DSV41_ENGRAM_ZERO.get():
            return x
        values = gathered
        if not self.embed._shared and self.embed.tp_size > 1:
            values = tensor_model_parallel_all_reduce(values)
        return self._wkv_gate(x, values)

    def project(
        self, hash_ids: torch.Tensor, *, cp_all_tokens: bool = False
    ) -> torch.Tensor:
        kv, _ = self.wkv(self.embed(hash_ids, cp_all_tokens=cp_all_tokens).flatten(-2))
        return kv

    def apply_gate(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        return engram_gate(
            x, kv, self.q_weight, self.k_weight, self.eps, self.clamp_value
        )
