"""Triton gather of DeepSeek-V4.1 engram rows: fp8 e4m3 payload, e8m0 block scales.

The table pointers arrive as raw addresses so one kernel serves a device table, a
pinned host table, or (on Grace-Blackwell, through ATS) a plain host mapping.

Dequant is fp32(row) * 2**(exp - 127) then a single round to the activation
dtype. Hopper/Blackwell keep bf16. SM70 has no BF16 and no Triton ``fp8e4nv``,
so Volta loads the payload as uint8, converts e4m3fn in software, and stores
fp16.
"""

import torch
import triton
import triton.language as tl

# e8m0 has no zero: the exponent byte 0 encodes 2**-127.
_E8M0_ZERO = 2.0**-127


@triton.jit
def _engram_gather_kernel(
    w_ptr,
    s_ptr,
    ids_ptr,
    out_ptr,
    row_lo,
    row_hi,
    DIM: tl.constexpr,
    BLK: tl.constexpr,
    E8M0_ZERO: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    # The table holds rows [row_lo, row_hi); an id outside it is not read and
    # comes out as zeros, which is what the sharded all-reduce sums.
    owned = (idx >= row_lo) & (idx < row_hi)
    local = tl.where(owned, idx - row_lo, 0)
    # SM70 Triton's fp8e4nv is unsupported; uint8 + software e4m3fn matches
    # torch.float() of float8_e4m3fn (including denorms: man * 2**-9).
    w = w_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    s = s_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    offs = tl.arange(0, DIM)
    raw = tl.load(w + local * DIM + offs, mask=owned, other=0).to(tl.int32)
    sign = (raw >> 7) & 1
    exp = (raw >> 3) & 0xF
    man = raw & 7
    denorm = man.to(tl.float32) * (2.0**-9)
    norm = (1.0 + man.to(tl.float32) * 0.125) * tl.math.exp2(exp.to(tl.float32) - 7.0)
    vals = tl.where(exp == 0, denorm, norm)
    vals = tl.where((exp == 0) & (man == 0), 0.0, vals)
    # e4m3fn: exp=15, man=7 is NaN (no Inf).
    vals = tl.where((exp == 15) & (man == 7), float("nan"), vals)
    vals = tl.where(sign == 1, -vals, vals)
    exps = tl.load(s + local * (DIM // BLK) + offs // BLK, mask=owned, other=0).to(
        tl.int32
    )
    # 2**(e - 127) from the exponent bits: exact, no exp2 rounding or denormal flush.
    scale = (exps << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exps == 0, E8M0_ZERO, scale)
    out = tl.where(owned, vals * scale, 0.0)
    if OUT_BF16:
        tl.store(out_ptr + row * DIM + offs, out.to(tl.bfloat16))
    else:
        tl.store(out_ptr + row * DIM + offs, out.to(tl.float16))


def _sm70_gather() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 7


def engram_gather(
    weight_ptr: int,
    scale_ptr: int,
    ids: torch.Tensor,
    out: torch.Tensor,
    dim: int,
    block_size: int,
    row_lo: int = 0,
    row_hi: int = 2**62,
) -> torch.Tensor:
    """Gather rows ``ids`` ([N] int) into ``out`` ([N, dim] fp16/bf16, contiguous).

    ``weight_ptr`` addresses [rows, dim] fp8 e4m3 bytes and ``scale_ptr``
    [rows, dim // block_size] e8m0 bytes for global rows [row_lo, row_hi); both
    may live in device or host memory. Ids outside the range produce zero rows.
    SM70 requires ``out.dtype == torch.float16``.
    """
    assert dim & (dim - 1) == 0 and dim % block_size == 0, (dim, block_size)
    assert out.is_contiguous()
    if _sm70_gather():
        assert out.dtype == torch.float16, (
            f"SM70 engram_gather writes fp16, not {out.dtype}"
        )
    else:
        assert out.dtype in (torch.bfloat16, torch.float16), out.dtype
    n = ids.numel()
    if n:
        _engram_gather_kernel[(n,)](
            weight_ptr,
            scale_ptr,
            ids,
            out,
            row_lo,
            row_hi,
            DIM=dim,
            BLK=block_size,
            E8M0_ZERO=_E8M0_ZERO,
            OUT_BF16=out.dtype == torch.bfloat16,
        )
    return out
