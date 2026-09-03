<div align="center" id="sglang-v100-top">

# SGLang&nbsp;V100

**Qwen3.8-Flash-Next at full 262K context on 4× NVIDIA V100 SXM2.**

A Volta (sm70) port of [SGLang](https://github.com/sgl-project/sglang).

</div>

---

## What this is

Upstream SGLang does not support Volta. Neither does anything else that can serve
a 125B mixture-of-experts model at long context: CUDA 13 dropped sm70 outright,
FlashAttention needs sm80+, and Volta has no bfloat16 at all.

This fork closes that gap. It serves **Qwen3.8-Flash-Next** — 125B MoE with a
51 GB host-offloaded PLE n-gram table, a hybrid 36×GDN + 12×QSA attention stack,
a built-in MTP draft head and a vision tower — on four 32 GB V100s, at the
model's native 262,144-token context, with NVFP4 weights and an FP8-E5M2 KV
cache.

If you have V100s sitting idle because modern inference stacks abandoned them,
this makes them useful again for frontier-class long-context agentic work.

## Measured performance

Single node, 4× V100-SXM2-32GB, TP=4, 131,508-token prompt + 300 generated
tokens, temperature 0, cold prefill (prefix cache flushed):

| | total | prefill | decode | MTP accept |
|---|---|---|---|---|
| `target` (no speculation) | **36.7 s** | **4,139 tok/s** | **60.4 tok/s** | — |
| `mtp` (EAGLE, 3 steps) | **38.9 s** | **3,861 tok/s** | **62.3 tok/s** | **0.57–0.65** |

Short-context decode with MTP runs 61–74 tok/s. Idle cost is ~4% CPU per rank
and 0% GPU — the scheduler blocks on a poller rather than spinning.

> For scale: on the same host and the same 131.5K input, `llama.cpp` in
> layer-split mode took 639–663 s. That is not a like-for-like comparison —
> llama.cpp's tensor-parallel mode was unavailable for this architecture, and
> layer-split serialises across GPUs — but it is the practical alternative on
> this hardware, and the gap is roughly 16×.

## Hardware and software requirements

| | |
|---|---|
| GPUs | 4× V100 32 GB (SXM2 recommended; NVLink helps, a partial mesh is fine) |
| Host RAM | **~134 GB measured in use** at 262K context with `--hicache-size 8`. 160 GB is a comfortable floor. The host cache tier scales with `--hicache-size`, so you can trade it down on a smaller box |
| Disk | 126 GB for the NVFP4 weights, plus space for the disk cache tier |
| CUDA | 12.8 or 12.9 — **not 13.x**, which removed Volta support |
| Host compiler | GCC **≤ 14** with a working `cc1plus`. CUDA 12.9 rejects GCC 15, and many distros now default to it |
| Python | 3.12 |

The 32 GB-per-GPU figure is not negotiable: the NVFP4 weights alone are ~22 GB
per rank at TP=4. The host-RAM and disk figures are measured on a running
system, not estimated.

## Quick start

```bash
git clone https://github.com/dg1kjd/sglang-v100-sxm2-qwen3.8-flash-next.git
cd sglang-v100-sxm2-qwen3.8-flash-next

# Full build: system deps, venv, patched FlashInfer, TurboMind, sglang-kernel,
# Marlin. Takes roughly an hour, most of it nvcc.
bash scripts/install_v100.sh

# Verify the SM70 stack registered correctly.
bash scripts/smoke_v100.sh
```

### Get the model

The validated checkpoint is the NVFP4 quantisation of Qwen3.8-Flash-Next:

```bash
pip install -U "huggingface_hub[cli]"
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --local-dir ~/models/Qwen3.8-Flash-Next-NVFP4

export FLASH_NEXT_MODEL=~/models/Qwen3.8-Flash-Next-NVFP4
```

126 GB. It is the multimodal export, so the vision tower comes with it — check
`language_model_only: false` in `config.json` if in doubt.

| | |
|---|---|
| checkpoint | [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) |
| base model | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| quantisation | NVFP4 W4A16 (modelopt), FP8-E5M2 KV cache at runtime |

Other checkpoints of the same architecture should work but are untested here.
The model is subject to its own license, which you must satisfy independently.

Step-by-step instructions, and what to do when a step fails, are in
**[docs/v100/INSTALL.md](docs/v100/INSTALL.md)**.

### Serving

```bash
# Long-context serving, no speculation
bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh target

# Same, plus the built-in MTP draft head (recommended)
bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp
```

The launcher carries the tuned flag set and, more usefully, the *reasons* for
each OOM-sensitive value in its comments. Read it before changing
`--mem-fraction-static`, `--max-prefill-tokens` or `--hicache-size`.

### Talking to it

Both API surfaces are native, not shims:

```bash
# OpenAI-compatible
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38next-nvfp4","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'

# Anthropic Messages API -- Claude Code connects to this directly
curl http://127.0.0.1:30000/v1/messages \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"qwen38next-nvfp4","max_tokens":128,"messages":[{"role":"user","content":"Hello"}]}'
```

Image input works on both. Note that this model emits reasoning: an empty
`content` alongside a large `completion_tokens` means the reply hit `max_tokens`
while still inside a thinking block — raise the limit rather than reading it as
a failure.

## What the port adds

Everything below is fork-owned; none of it exists upstream.

- **NVFP4 W4A16 on sm70** — a JIT CUDA path for FP4 weights on hardware with no
  FP4 support, plus the Marlin V100 GPTQ/AWQ repack kernels.
- **TileLang attention for Volta** (`tilelang_fa_v100`) — paged prefill, decode
  and verify kernels, registered as a first-class attention backend.
- **QSA sparse attention** with a compressed index cache, and its own KV pool
  (`QSATokenToKVPool`) carrying the compressed-key buffers.
- **GDN linear attention** in TileLang and Triton, tuned for sm70 occupancy.
- **TurboMind sm70 backend** for block-FP8 and FP16 MoE, plus an exact AWQ
  dequantiser.
- **FP8-E5M2 KV cache** on hardware without native FP8.
- **PLE host offload** — the 51 GB n-gram table lives in host RAM, with the
  per-request n-gram and short-conv state riding the mamba slot lifecycle.
- **Single-stage custom all-reduce**, because two-stage is pathological on a
  partial NVLink mesh.
- **fp16 forcing** throughout, since Volta has no bf16 (`SGLANG_SM70_FORCE_FP16`).

## Limitations and known gaps

Stated plainly, because the alternative is you finding them at 3am:

- **Only Qwen3.8-Flash-Next is validated.** Other architectures may load; none
  are tested here, and several upstream model paths assume sm80+ kernels.
- **`multimodal_gen` (diffusion / video generation) is not ported.** It carries
  upstream's code, not this fork's Volta adaptations. The Qwen3.8 *vision tower*
  is fully working — that is a different subsystem.
- **No 24-hour soak has been run** on the current tree. A 24-request mixed
  workload shows no leak or instability, which is not the same thing.
- **A cold FlashInfer JIT cache costs several minutes** on first launch, and
  four TP ranks will compile in parallel. Subsequent launches are fast.
- **The dense NVFP4 linear path is unverified.** It matters only if a checkpoint
  quantises weights outside the MoE experts; Qwen3.8-Flash-Next does not.

Open items are tracked in [`docs/v100/KNOWN-ISSUES.md`](docs/v100/KNOWN-ISSUES.md).

## Relationship to upstream

This is a fork of [sgl-project/sglang](https://github.com/sgl-project/sglang),
re-based onto upstream `main` as of 2026-09-02 (`99b910955`). Upstream's engine
— including the unified radix cache, the hierarchical KV cache and the
speculative decoding stack — is used as-is wherever possible; this fork adds the
sm70 layer and the Qwen3.8-Flash-Next model support on top.

Every deviation from upstream carries its reasoning in the commit that made it;
`git log` is the record. The procedure for taking a newer upstream — including
the one trap that matters — is in
[`docs/v100/UPSTREAM-SYNC.md`](docs/v100/UPSTREAM-SYNC.md).

Bug reports about the sm70 path belong here. Bug reports about SGLang itself
belong upstream.

## Credits

This is a derivative work of [SGLang](https://github.com/sgl-project/sglang)
(Apache 2.0, Copyright 2023-2024 SGLang Team). Upstream does the hard part; this
fork adds a Volta layer on top.

The Volta build also stands on
[marlin_v100](https://github.com/zhinianqin/marlin_v100),
[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) (TurboMind sm70),
[CUTLASS](https://github.com/NVIDIA/cutlass),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer) and
[TileLang](https://github.com/tile-ai/tilelang). None are redistributed here —
the build fetches them at pinned revisions. Full attribution in
[NOTICE](NOTICE).

## License and disclaimer

Apache 2.0, inherited from SGLang — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

This is an **independent community fork**. It is not affiliated with, endorsed
by, or supported by the SGLang project, LMSYS, NVIDIA, or the model's authors.

Provided **as is, without warranty or condition of any kind**, per Section 7 of
the Apache License. It drives hardware its vendor no longer supports, using
kernels written specifically for that purpose; validate it in your own
environment before relying on it for anything that matters.

No model weights are distributed here. Any checkpoint you use remains subject to
its own license and terms, which you must satisfy independently.

## Contact

Issues and pull requests are the preferred channel. For anything that does not
belong in public, `git@jens-david-consulting.com`.
