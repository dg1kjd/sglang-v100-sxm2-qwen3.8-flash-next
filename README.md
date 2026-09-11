<div align="center" id="sglang-v100-top">

# SGLang&nbsp;V100

*"Cool-kids-on-steroids"-Release ;)*

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
model's native 262,144-token context, with NVFP4 weights and an FP16 KV cache.

If you have V100s sitting idle because modern inference stacks abandoned them,
this makes them useful again for frontier-class long-context agentic work.

## Measured performance

Single node, 4× V100-SXM2-32GB, TP=4, the built-in MTP draft head on,
`--mem-fraction-static 0.86`, warm JIT. Greedy (temperature 0), thinking
disabled, cold prefill (prefix cache flushed before each request).

**Decode.** Measured two ways, because the workloads a reader cares about are
different:

| workload | decode (tok/s) | MTP accept len |
|---|---|---|
| Coding problems — 8× HumanEval, the 1Cat-vLLM comparison | **156.6** (median, 150–160) | 3.90 |
| Agentic long context — 7,413-token prompt, one stream | **127** | ~3.3 |

Agentic decode under concurrency — per-stream median over 194–289 requests in a
~1-hour sustained load:

| concurrency | generation (tok/s, per stream) | time to first token |
|---|---|---|
| 1 | **127** | 2.96 s |
| 2 | **78.9** | 4.74 s |
| 4 | **54.0** | 8.70 s |

Per-stream is what a single request sees; aggregate still climbs with
concurrency (four streams ≈ 216 tok/s combined).

**Prefill** scales with prompt length — fixed per-request overhead dominates
short prompts and amortises over long ones:

| prompt tokens | 375 | 1,473 | 2,936 | 5,862 | 11,714 | 23,417 |
|---|---|---|---|---|---|---|
| prefill tok/s | 283 | 986 | 1,746 | 2,395 | 2,888 | 3,092 |

The 7,413-token agentic prompt prefills at ~2,560 tok/s (≈2.9 s cold). The
262K context is real: a ~131,500-token prompt prefills in ~48 s (~2,700 tok/s).
Idle cost is ~4% CPU per rank and 0% GPU — the scheduler blocks on a poller
rather than spinning.

> For scale: on the same host, `llama.cpp` in layer-split mode took 639–663 s
> for the same ~131.5K-token prompt. Not a like-for-like comparison — llama.cpp's
> tensor-parallel mode was unavailable for this architecture, and layer-split
> serialises across GPUs — but it is the practical alternative on this hardware,
> and it is roughly an order of magnitude slower.

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

`install_v100.sh` (the build) plus `smoke_v100.sh` (the check) is the entire
install; **[docs/v100/INSTALL.md](docs/v100/INSTALL.md)** documents what each
step does and what to do when one fails. **Do not skip the smoke check** — no
prebuilt kernels are distributed (the `.so` files are build outputs), and the
stock Marlin MoE kernel is an empty stub below sm80, so a server missing the
V100 kernels starts, answers, and returns zero-valued expert output: confident
nonsense rather than an error.

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
| quantisation | NVFP4 W4A16 (modelopt), FP16 KV cache at runtime |

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

None of this exists upstream. **The Volta port itself is the work of
[haohervchb](https://github.com/haohervchb/sglang-V100)** — the sm70 kernels, the model support and the serving
recipe below are all theirs. This repository re-lands that work onto a much
newer SGLang and fixes what the move broke; see
[Relationship to upstream](#relationship-to-upstream).

- **NVFP4 W4A16 on sm70** — a JIT CUDA path for FP4 weights on hardware with no
  FP4 support, plus the Marlin V100 GPTQ/AWQ repack kernels.
- **TileLang attention for Volta** (`tilelang_fa_v100`) — paged prefill, decode
  and verify kernels, registered as a first-class attention backend.
- **QSA sparse attention** with a compressed index cache, and its own KV pool
  (`QSATokenToKVPool`) carrying the compressed-key buffers.
- **GDN linear attention** in TileLang and Triton, tuned for sm70 occupancy.
- **TurboMind sm70 backend** for block-FP8 and FP16 MoE, plus an exact AWQ
  dequantiser.
- **FP16 / FP8-E5M2 KV cache** on hardware without native FP8. The FP16 path —
  the production dtype — uses a fast sm70 sparse decode kernel that reads only
  the selected top-k K/V, so its higher per-token precision costs no decode
  speed on this sparse-attention model.
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
- **Stability was hammered, not soaked.** A ~1-hour sustained load — agentic
  prompts at np 1/2/4 plus a beyond-spec 32k-token / np 8 phase — ran with no
  crash and no incorrect output at the current `--mem-fraction-static 0.86`. It
  did surface one prefill OOM at the previous 0.88 default under the beyond-spec
  load; the 0.86 retune fixed it (rationale in the serve-script comment). A
  multi-day soak has not been run.
- **Greedy output is not bit-reproducible across cache states.** A property of
  the FP16 mamba-hybrid pipeline with a radix cache, not a defect: the cache
  replays an approximate GDN (linear-attention) state for a cached prefix, so a
  prompt's exact tokens can differ a little between a cold and a warm prefix,
  and prompts sitting on a token decision boundary can vary across runs. Every
  output is a valid completion — no corruption or garbage.
- **A cold FlashInfer JIT cache costs several minutes** on first launch, and
  four TP ranks will compile in parallel. Subsequent launches are fast.
- **The dense NVFP4 linear path is unverified.** It matters only if a checkpoint
  quantises weights outside the MoE experts; Qwen3.8-Flash-Next does not.

## Relationship to upstream

This is a downstream of [haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100), which is itself a fork
of [sgl-project/sglang](https://github.com/sgl-project/sglang). The V100 port
was cut from upstream around 2026-06-01 and had not been re-synced since; this
repository re-lands it onto upstream `main` as of 2026-09-02 (`99b910955`),
about 4,250 commits later. Upstream's engine
— including the unified radix cache, the hierarchical KV cache and the
speculative decoding stack — is used as-is wherever possible; this fork adds the
sm70 layer and the Qwen3.8-Flash-Next model support on top.

This is not a pure 3-way merge between the two upstream repos. Beyond re-landing
the port, the tree carries hand-crafted optimizations and bug fixes, and it is
ruggedized, tested, and plug-and-play — it runs as shipped. It is also ongoing:
we intend to keep pulling in upstream improvements as well as continuing our own
work on top.

Every deviation from upstream carries its reasoning in the commit that made it;
`git log` is the record.

Bug reports about the sm70 path belong here. Bug reports about SGLang itself
belong upstream.

## Credits

**The Volta port is [haohervchb](https://github.com/haohervchb/sglang-V100)'s work.** Every sm70 kernel in here —
the TileLang attention backend, QSA, the GDN linear-attention kernels, NVFP4 on
hardware with no FP4 support, the TurboMind sm70 backend, the PLE host offload,
the Qwen4-Exp model support — was written there, along with the serving recipe
and the tuning that makes it fit in 32 GB. If this is useful to you, that is
where the credit belongs. The patched sm70 FlashInfer the build uses is also
theirs ([haohervchb/flashinfer](https://github.com/haohervchb/flashinfer)).

This repository's contribution is narrower: re-landing that port onto an SGLang
roughly 4,250 commits newer, and fixing what the move broke.

Both are derivative works of [SGLang](https://github.com/sgl-project/sglang)
(Apache 2.0, Copyright 2023-2024 SGLang Team), which does the hard part.

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
