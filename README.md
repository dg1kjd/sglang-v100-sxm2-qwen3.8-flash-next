<div align="center" id="sglang-v100-top">

# SGLang&nbsp;V100

*"Cool-kids-on-more-steroids"-Release ;)*

**Qwen3.8-Flash-Next at full 262K context on 4× NVIDIA V100 SXM2.**

Initial **DeepSeek-V4.1-Flash** serve on 8× V100 (DSpark, 256k).

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

It also has an **initial** serve path for official
**DeepSeek-V4.1-Flash** (https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
on eight 32 GB V100s: CSA2 sparse attention, host Engram, MXFP4 expert spill,
and the checkpoint's own DSpark draft. That recipe is new and not soaked the
way Qwen is — see [DeepSeek-V4.1-Flash](#deepseek-v41-flash).

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
| Coding problems — 8× HumanEval, the 1Cat-vLLM comparison | **167.7** (median, 153–177) | 3.57 |
| Agentic long context — 7,405-token prompt, one stream | **138** | 3.0 |

Agentic decode under concurrency — per-stream median over 8 requests at each
concurrency (live `--max-running-requests 3`):

| concurrency | generation (tok/s, per stream) | time to first token |
|---|---|---|
| 1 | **138** | 1.95 s |
| 2 | **80.3** | 3.03 s |
| 3 | **64.3** | 2.96 s |

Per-stream is what a single request sees; aggregate still climbs with
concurrency (three streams ≈ 193 tok/s combined).

**Prefill** scales with prompt length — fixed per-request overhead dominates
short prompts and amortises over long ones:

| prompt tokens | 375 | 1,473 | 2,936 | 5,862 | 11,714 | 23,417 |
|---|---|---|---|---|---|---|
| prefill tok/s | 1,394 | 1,418 | 2,488 | 3,451 | 3,258 | 3,266 |

The 7,405-token agentic prompt prefills at ~3,060 tok/s (≈2.4 s cold). The
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

# DeepSeek-V4.1-Flash (initial, 8×V100) — see recipe below
bash scripts/serve_dsv41_v100.sh
```

The Qwen launcher carries the tuned flag set and, more usefully, the *reasons*
for each OOM-sensitive value in its comments. Read it before changing
`--mem-fraction-static`, `--max-prefill-tokens` or `--hicache-size`.
On V100 (`--dtype float16`) it also passes `--ple-offload-embedding`, so the
51 GB PLE n-gram table lands in host memory. Without that offload the table
is created on GPU and OOMs at load.

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

## DeepSeek-V4.1-Flash

Initial attempt: official `deepseek-ai/DeepSeek-V4.1-Flash` on **8×**
V100-SXM2-32GB (TP=8 / EP=8). Different box shape than Qwen (four cards is not
enough). Host Engram (~189 GiB) plus pinned expert spill need a large RAM node
next to the GPUs and **1G hugepages** on that NUMA node; see
`scripts/serve_dsv41_v100.sh` and [docs/v100/INSTALL.md](docs/v100/INSTALL.md).

### Get the model

Use the official DeepSeek mixed-quant checkpoint — not a third-party NVFP4 /
GPTQ / AWQ re-quant, and not DeepSeek-V4-Flash (that is a different
architecture). Dense weights are block FP8 (`quant_method: fp8`, 32×32
`ue8m0`); routed experts are native FP4 (`expert_dtype: fp4`, MXFP4). The
DSpark draft lives in the same repo. Runtime on this port is FP16 activations
and an FP8-E4M3 KV cache; do not unpack the dense MXFP8 to FP16 on 32 GB.

Weights: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash

```bash
pip install -U "huggingface_hub[cli]"
hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --local-dir ~/models/DeepSeek-V4.1-Flash

export MODEL_PATH=~/models/DeepSeek-V4.1-Flash
```

~476 GB (48 shards). The export is multimodal; this recipe serves it with
`--language-model-only`. The model is MIT-licensed; satisfy that independently.

| | |
|---|---|
| checkpoint | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash |
| quantisation | native mixed: MXFP8 dense (e4m3 + UE8M0, 32×32) + MXFP4 routed experts |
| do not use | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash , https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 , https://huggingface.co/nvidia/DeepSeek-V4-Flash-nvfp4-DSpark (V4-Flash NVFP4, not V4.1) |

Measured on that 8-card recipe (DSpark on, sticky last-seq, advertised 256k
context, `np=1`). Coding and short-prompt checks at temperature 0; 8k prefill
with `max_new_tokens=1`:

| workload | result |
|---|---|
| `17*19` / `is_palindrome` | correct |
| coding-1 (`merge_sorted`, 101 tok) | **8.14 tok/s** |
| 8k prefill | **7.98 s / 1026 tok/s** |

Open-ended chat at temperature 1 is closer to **3 tok/s** (DSpark accept ~2).
Temperature 0 is right for code and wrong for long prose: greedy can lock onto
a short cycle and the target will keep signing it. Use `temperature=1`,
`top_p=0.95` for chat. Do not call `/health` (it drops the sticky pin).

### Reference recipe

The wrapper is the supported entry. It exports the env knobs that are not CLI
flags, then launches the server.

```bash
export MODEL_PATH=~/models/DeepSeek-V4.1-Flash
export SGLANG_DSV41_DSPARK=1
export SGLANG_DSV41_STICKY_LAST_SEQ=1
export SGLANG_DSV41_CONTEXT_LEN=262144
bash scripts/serve_dsv41_v100.sh
```

Expanded (what the script actually runs when DSpark is on). Do not drop the
env block — spill, Engram, and sticky are not implied by the CLI flags.

```bash
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=private
export SGLANG_DSV41_EXPERT_SPILL_APPLY=1
export SGLANG_DSV41_EXPERT_SPILL_GB=13
export SGLANG_DSV41_SPILL_LANDING=36
export SGLANG_DSV41_STICKY_LAST_SEQ=1
export NCCL_ALGO=allreduce:tree
export NCCL_BUFFSIZE=2097152
export NCCL_MIN_NCHANNELS=1
export NCCL_MAX_NCHANNELS=4

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tp 8 --ep-size 8 \
  --dtype float16 \
  --moe-runner-backend marlin \
  --attention-backend dsv4 \
  --context-length 262144 \
  --chunked-prefill-size 2048 \
  --mem-fraction-static 0.88 \
  --max-running-requests 1 \
  --max-total-tokens 262144 \
  --max-prefill-tokens 262144 \
  --pre-warm-nccl \
  --disable-prefill-cuda-graph \
  --cuda-graph-max-bs-decode 1 \
  --disable-radix-cache \
  --language-model-only \
  --reasoning-parser deepseek-v41 \
  --tool-call-parser deepseekv41 \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 30000
```

| knob | ship value | why |
|---|---|---|
| DSpark | on (`γ=5` from the checkpoint) | Best measured TG on this box. `SGLANG_DSV41_DSPARK=0` is greedy Tree |
| sticky last-seq | on | Exact full-history continuation only. Radix stays off (CSA2 rings desync on a prefix hit) |
| context / max tokens | 262144 | Advertised window. 8k prefill is what has been smoked; 512k has not left ~300 MiB for the Engram MXFP8 unpack |
| `--mem-fraction-static` | 0.88 | 0.99 OOMs the Engram unpack on T=6 verify capture / 8k prefill |
| expert spill | 13 GiB/rank (landing 36) | Spill 12 left 8k ~8 MiB short of that unpack |
| `--max-running-requests` | 1 | DSpark would otherwise inflate this |
| `--chunked-prefill-size` | 2048 | Vestigial SWA floor is sized for this chunk |

Do not tune `--speculative-dspark-block-size`. Checkpoint weights stay mixed
MXFP4 experts + packed MXFP8 dense; do not unpack dense to FP16 on 32 GB.

## What the port adds

None of this exists upstream. **The Volta port itself is the work of
[haohervchb](https://github.com/haohervchb/sglang-V100)** — the sm70 kernels, the
Qwen3.8 model support and that serving recipe are all theirs. This repository
re-lands that work onto a much newer SGLang, fixes what the move broke, and
adds the DeepSeek-V4.1-Flash path; see
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

- **Qwen3.8-Flash-Next is the soaked model.** DeepSeek-V4.1-Flash has an
  initial 8×V100 recipe (8k prefill + coding smoke). It is not a multi-hour
  soak, leftover HBM after an 8k prompt is tight (~0.5 GiB), and open-ended
  greedy (temperature 0) can loop. Other architectures may load; several
  upstream model paths still assume sm80+ kernels.
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
sm70 layer, Qwen3.8-Flash-Next, and an initial DeepSeek-V4.1-Flash serve path
on top.

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
