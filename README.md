<div align="center" id="sglang-v100-top">

# SGLang&nbsp;V100

*"Cool-kids-on-more-steroids"-Release ;)*

**Qwen3.8-Flash-Next** (125B MoE, NVFP4 W4A16, FP16 KV, 4× V100-32GB, 262k) — prefill ~3,000 tok/s, decode ~100 tok/s per stream, ~180 tok/s aggregate at 3 streams.

**DeepSeek-V4.1-Flash** (official mixed MXFP8+MXFP4, DSpark, 8× V100-32GB, 256k, one stream) — short code ~9 tok/s, warm prefill ~560 tok/s.

A Volta (sm70) port of [SGLang](https://github.com/sgl-project/sglang). Those two models are the supported ones. Others may load; they are untested here.

</div>

---

## What this is

Upstream SGLang does not support Volta. CUDA 13 dropped sm70, FlashAttention needs sm80+, and Volta has no bfloat16. This fork serves frontier-class long-context models on V100s anyway, including agentic coding through the native Anthropic Messages API (Claude Code connects directly).

**Qwen3.8-Flash-Next** is the soaked model: 125B MoE, a 51 GB host-offloaded PLE n-gram table, hybrid 36×GDN + 12×QSA attention, a built-in MTP draft head, and a vision tower. It runs at the model's native 262,144-token context on four 32 GB V100s, NVFP4 weights, FP16 KV. On the MTP recipe below, prefill holds about **3,000 tok/s** from 8k through 128k. One stream decodes at about **100 tok/s** (~46 target forwards/s, accept length ~2.1 on this padding workload). Three streams reach about **180 tok/s** aggregate. Full table: [Qwen](#qwen38-flash-next).

**DeepSeek-V4.1-Flash** ([checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)) is the second engine, and the one this tree serves on Volta: CSA2 sparse attention, host Engram, MXFP4 expert spill, and the checkpoint's own DSpark draft, on eight 32 GB V100s. One resident agent session continues from a recorded prefix stop. A second session still prefills from scratch. Short code is about **9 tok/s**; warm 8k prefill is about **560 tok/s**. Full table: [DeepSeek-V4.1-Flash](#deepseek-v41-flash).

## Hardware and software requirements

| | Qwen3.8-Flash-Next | DeepSeek-V4.1-Flash |
|---|---|---|
| GPUs | 4× V100 32 GB (SXM2 recommended; NVLink helps, a partial mesh is fine). Four cards are the Qwen shape | **8×** V100-SXM2-32GB, TP=8 / EP=8. Four cards are not enough |
| Host RAM | **~134 GB measured in use** at 262k with `--hicache-size 8`. 160 GB is a comfortable floor. The host cache tier scales with `--hicache-size` | Host Engram (~189 GiB) plus pinned expert spill, on a large RAM node next to the GPUs, with **1G hugepages** on that NUMA node |
| Disk | 126 GB NVFP4 weights, plus the disk cache tier | ~476 GB (48 shards) |
| Context | 262,144 | 262,144 advertised. 8k prefill is what has been smoked; 512k has not left ~300 MiB for the Engram MXFP8 unpack |
| CUDA | 12.8 or 12.9. CUDA 13.x removed Volta | same |
| Host compiler | GCC **≤ 14** with a working `cc1plus`. CUDA 12.9 rejects GCC 15, and many distros now default to it | same |
| Python | 3.12 | 3.12 |

The 32 GB-per-GPU figure is not negotiable for Qwen: the NVFP4 weights alone are ~22 GB per rank at TP=4. The host-RAM and disk figures are measured on a running system. Four PCIe-only V100s (P2P, no NVLink): set `SGLANG_CUSTOM_AR_ALLOW_PCIE=1` and `NCCL_P2P_LEVEL=PXB`. Those stay off by default; leave them off on an 8× hybrid NVLink mesh.

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

`install_v100.sh` plus `smoke_v100.sh` is the entire install. **[docs/v100/INSTALL.md](docs/v100/INSTALL.md)** documents each step and what to do when one fails. Run the smoke check. No prebuilt kernels are distributed (the `.so` files are build outputs), and the stock Marlin MoE kernel is an empty stub below sm80, so a server missing the V100 kernels starts, answers, and returns zero-valued expert output.

One engine at a time. Qwen and DeepSeek bind the same address, `0.0.0.0:11435` (`SGLANG_V100_HOST` / `SGLANG_V100_PORT`), not port 30000.

```bash
# Qwen, long context, no speculation
bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh target

# Qwen plus the built-in MTP draft head (recommended; the numbers below)
bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp

# DeepSeek-V4.1-Flash, 8× V100
bash scripts/serve_dsv41_v100.sh
```

## Qwen3.8-Flash-Next

### Get the model

The validated checkpoint is the NVFP4 quantisation of Qwen3.8-Flash-Next. 126 GB. It is the multimodal export, so the vision tower comes with it. `language_model_only: false` in `config.json` is the check. Other checkpoints of the same architecture may work; they are untested. The model stays under its own license.

```bash
pip install -U "huggingface_hub[cli]"
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --local-dir ~/models/Qwen3.8-Flash-Next-NVFP4

export FLASH_NEXT_MODEL=~/models/Qwen3.8-Flash-Next-NVFP4
```

| | |
|---|---|
| checkpoint | [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) |
| base model | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| quantisation | NVFP4 W4A16 (modelopt), FP16 KV cache at runtime |

The launcher comments record why each OOM-sensitive flag is what it is. Read them before changing `--mem-fraction-static`, `--max-prefill-tokens`, or `--hicache-size`. On V100 (`--dtype float16`) the script passes `--ple-offload-embedding`, so the 51 GB PLE n-gram table lands in host memory. Without that offload the table is created on GPU and OOMs at load.

### Talking to it

Both API surfaces are native.

```bash
# OpenAI-compatible
curl http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38next-nvfp4","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'

# Anthropic Messages API — Claude Code connects to this directly
curl http://127.0.0.1:11435/v1/messages \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"qwen38next-nvfp4","max_tokens":128,"messages":[{"role":"user","content":"Hello"}]}'
```

Image input works on both APIs. This model emits reasoning: an empty `content` alongside a large `completion_tokens` means the reply hit `max_tokens` while still inside a thinking block. Raise the limit.

### Measured performance

2026-09-24, `llm-decode-bench` 0.6.2, temperature 0, `scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp`. 4× V100-SXM2-32GB, TP=4, built-in MTP (3 steps / 4 draft tokens), `--mem-fraction-static 0.86`, `--max-running-requests 3`. This model's chat template leaves thinking on unless the request disables it, and the bench does not.

**Prefill.** Client prompt tokens / TTFT, one scout each. The 8k row also matched the server counter (3,065 tok/s).

| prompt tokens | TTFT | tok/s |
|---|---:|---:|
| 8,196 | 2.75 s | 2,977 |
| 32,150 | 10.53 s | 3,054 |
| 128,020 | 40.69 s | 3,146 |

**Decode.** Aggregate tok/s, then per stream. Accept length is tokens per target forward. At one stream the engine is about 46 forwards/s; the ~2.1 accept length is this padding workload. The padding cells are 15 s of `ignore_eos`. The coding row is the same tool, one short Python prompt, 4 runs, a 256-token cap: thinking stayed on and every run hit the cap.

| workload | C=1 | C=2 aggregate / per stream | C=3 aggregate / per stream |
|---|---:|---:|---:|
| padding, context 0 | **98.9** (accept 2.11) | **120.3** / 60.1 (2.02) | **177.9** / 59.3 (2.19) |
| padding, context 8k | **100.4** (2.18) | **118.9** / 59.5 (2.12) | **150.2** / 50.1 (2.10) |
| coding peak, 256-token cap | **119** median (117–119) | | |

Concurrency 4 is absent because `--max-running-requests 3` drops it. Aggregate still climbs through C=3.

### Reference recipe

The wrapper is the supported entry. `mtp` is the recipe the numbers above were measured on.

```bash
export FLASH_NEXT_MODEL=~/models/Qwen3.8-Flash-Next-NVFP4
bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp
```

Expanded (what that script runs for `mtp`). The env block is not implied by the CLI flags. `SGLANG_NUMA_BIND_V2=0` keeps the host cache interleaved; the file tier must sit on a real disk, not a tmpfs `/tmp`.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_LEVEL=NVL
export SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage
export SGLANG_MAMBA_CONV_DTYPE=float16
export SGLANG_MAMBA_SSM_DTYPE=float16
export SGLANG_SM70_FORCE_FP16=1
export SGLANG_SM70_DENSE_GEMV=1
export SGLANG_SM70_QWEN_FUSIONS=1
export SGLANG_NUMA_BIND_V2=0
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=$HOME/hicache_storage

python -m sglang.launch_server \
  --model-path "${FLASH_NEXT_MODEL}" \
  --served-model-name qwen38next-nvfp4 \
  --trust-remote-code \
  --dtype float16 \
  --quantization modelopt_fp4 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --attention-backend tilelang_fa_v100 \
  --linear-attn-prefill-backend tilelang \
  --linear-attn-decode-backend triton \
  --kv-cache-dtype auto \
  --tensor-parallel-size 4 \
  --context-length 262144 \
  --mem-fraction-static 0.86 \
  --max-running-requests 3 \
  --max-mamba-cache-size 20 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 4096 \
  --enable-hierarchical-cache \
  --hicache-size 8 \
  --hicache-mem-layout page_first \
  --hicache-storage-backend file \
  --hicache-write-policy write_back \
  --sleep-on-idle \
  --cuda-graph-max-bs-decode 3 \
  --cuda-graph-bs-decode 1 2 3 \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-full-memory-ratio 0.2 \
  --enable-cache-report \
  --enable-metrics \
  --ple-offload-embedding \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path "${FLASH_NEXT_MODEL}" \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --host 0.0.0.0 \
  --port 11435
```

| knob | ship value | why |
|---|---|---|
| MTP | on, 3 steps / 4 draft tokens | The measured recipe. `target` is the same server with no draft head |
| `--kv-cache-dtype` | `auto` (FP16 on V100) | The fast sm70 decode kernel reads only the selected top-k, so FP16 costs no decode speed and accepts better than `fp8_e5m2`. The 1-byte cache is `FLASH_NEXT_EXTRA_ARGS='--kv-cache-dtype fp8_e5m2'` plus a lower mem-fraction |
| `--mem-fraction-static` | 0.86 | 0.88 OOMed prefill under a beyond-spec 32k / np 8 load. 0.86 costs about 12% of the KV pool and still holds long agent contexts at 3 streams |
| `--max-running-requests` | 3 | Aggregate decode peaks here. A fourth stream adds no throughput and stretches TTFT. CUDA graphs are captured for batch 1–3 only |
| prefill chunk | 4096 | An 8k GDN chunk OOMed in the headroom left after the KV pool. Long prompts still run, in 4k chunks |
| `--hicache-size` | 8 per rank | Host KV and host Mamba, so total is `N × 2 × 4` ranks = 64 GB. N=16 left the box near thrashing. `page_first` is required for the Mamba host pool |
| write policy | `write_back` | `write_through` saturated the disk and made prefix readback collapse. Disk writes happen on eviction and shutdown |
| `--ple-offload-embedding` | on | The 51 GB n-gram table stays in host RAM. On GPU it OOMs at load |
| `--sleep-on-idle` | on | Without it each rank busy-spins a core while idle |
| `--enable-cache-report` | on | Surfaces prefix-cache hits to Claude Code. The cache already hits without the flag; the flag only reports them |

`target` drops the four `--speculative-*` lines. Leave `SGLANG_CUSTOM_AR_ALLOW_PCIE` unset on an NVLink mesh.

## DeepSeek-V4.1-Flash

Official `deepseek-ai/DeepSeek-V4.1-Flash` on 8× V100-SXM2-32GB. This snapshot has carried a multi-hour Claude Code session on a single conversation. Continuations that match a recorded chunk or request stop are not re-prefilled (a few dozen new tokens is a few seconds). A suffix of several thousand tokens that was never computed is still about 60 tok/s: each rank holds 30 experts on the GPU and a chunk often touches more, so the spill cache thrashes inside the chunk. Each rank's weight load reads only the experts it owns, and the draft reads only `mtp.*`. On this 8-card shape that was 1347 s for the target and 17 s for the draft, about 24 minutes until the server was ready (previously about 36 minutes, almost all of it weight load). Hugepages and the RAM layout are in `scripts/serve_dsv41_v100.sh` and [docs/v100/INSTALL.md](docs/v100/INSTALL.md).

### Get the model

Use the official DeepSeek mixed-quant checkpoint. Dense weights are block FP8 (`quant_method: fp8`, 32×32 `ue8m0`); routed experts are native FP4 (`expert_dtype: fp4`, MXFP4). The DSpark draft lives in the same repo. Runtime on this port is FP16 activations and an FP8-E4M3 KV cache. Leave the dense MXFP8 packed; unpacking it to FP16 does not fit in 32 GB. Skip third-party NVFP4 / GPTQ / AWQ re-quants, and skip DeepSeek-V4-Flash: that is a different architecture.

Weights: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash

```bash
pip install -U "huggingface_hub[cli]"
hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --local-dir ~/models/DeepSeek-V4.1-Flash

export MODEL_PATH=~/models/DeepSeek-V4.1-Flash
```

~476 GB (48 shards). The export is multimodal. The vision tower is one fp32 copy on rank 0; each GEMM streams through that GPU as fp16 and is dropped. The model is MIT-licensed.

| | |
|---|---|
| checkpoint | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash |
| quantisation | native mixed: MXFP8 dense (e4m3 + UE8M0, 32×32) + MXFP4 routed experts |
| do not use | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash , https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 , https://huggingface.co/nvidia/DeepSeek-V4-Flash-nvfp4-DSpark (V4-Flash NVFP4, not V4.1) |

### Measured performance

2026-09-24, `llm-decode-bench` 0.6.2, the 8-card recipe (DSpark on, sticky last-seq, radix cache off, advertised 256k, `np=1`). Temperature 0. The server was not started with `--enable-metrics`, so this run has no speculative-accept gauge. The ship script pins `--max-running-requests 1`.

| check | result |
|---|---|
| Coding peak, `merge_sorted`, natural stop at 104 tokens, 3 runs | **9.3** tok/s median (8.8–9.3) |
| Sustained padding decode, 20 s, `ignore_eos` | **2.8** tok/s (ITL 322 ms) |
| Cold prefill scout, server counted 5,286 prompt tokens | TTFT **45.3 s**, **117** tok/s |
| Warmer one-token prefill, 8,004 tokens, spill already touched | TTFT **14.3 s**, **558** tok/s |

The 2.8 tok/s cell is greedy padding, the case this model loops on. It is a different number from the 9.3 tok/s coding rate and from the ~60 tok/s uncached suffix above. The 117 tok/s scout is the cold first touch; 558 tok/s is the same box after that spill was already warm. Temperature 0 is right for short code and wrong for long prose. Use `temperature=1`, `top_p=0.95` for chat. The ship script leaves `/health` as a liveness probe (no generation), so a load balancer GET does not drop the sticky pin.

### Reference recipe

The wrapper is the supported entry. It exports the env knobs that are not CLI flags, then launches the server.

```bash
export MODEL_PATH=~/models/DeepSeek-V4.1-Flash
export SGLANG_DSV41_DSPARK=1
export SGLANG_DSV41_STICKY_LAST_SEQ=1
export SGLANG_DSV41_CONTEXT_LEN=262144
bash scripts/serve_dsv41_v100.sh
```

Expanded (what the script actually runs when DSpark is on). Spill, Engram, and sticky are set by the env block, not implied by the CLI flags.

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
  --mem-fraction-static 0.87 \
  --max-running-requests 1 \
  --max-total-tokens 262144 \
  --max-prefill-tokens 262144 \
  --pre-warm-nccl \
  --disable-prefill-cuda-graph \
  --cuda-graph-max-bs-decode 1 \
  --disable-radix-cache \
  --reasoning-parser deepseek-v41 \
  --tool-call-parser deepseekv41 \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 11435
```

| knob | ship value | why |
|---|---|---|
| DSpark | on (`γ=5` from the checkpoint) | Best measured TG on this box. `SGLANG_DSV41_DSPARK=0` is greedy Tree |
| sticky last-seq | on | One resident conversation. Exact continuation, or a shorter prefix saved at a chunk or request stop. Anything else, including a second conversation, drops the pin and prefills from 0. Radix stays off (a radix hit desyncs CSA2 pending state) |
| context / max tokens | 262144 | Advertised window. 8k prefill is what has been smoked; 512k has not left ~300 MiB for the Engram MXFP8 unpack |
| `--mem-fraction-static` | 0.87 | 0.99 OOMs the Engram unpack on T=6 verify capture. 0.88 OOMed the same 300 MiB unpack on a 461-token sticky prefill (TP7 had 284 MiB). 0.86 raises: no KV pool after draft weights |
| expert spill | 13 GiB/rank (landing 36) | Spill 12 left 8k ~8 MiB short of that unpack |
| `--max-running-requests` | 1 | DSpark would otherwise inflate this |
| `--chunked-prefill-size` | 2048 | Vestigial SWA floor is sized for this chunk |

Leave `--speculative-dspark-block-size` at the checkpoint default. Checkpoint weights stay mixed MXFP4 experts + packed MXFP8 dense.

## What the port adds

None of this exists upstream. The Volta port itself — sm70 kernels, Qwen3.8 model support, and that serving recipe — is [haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100). This repository re-lands that work onto a much newer SGLang, fixes what the move broke, and adds the DeepSeek-V4.1-Flash path. Credit and lineage are at the bottom.

- **NVFP4 W4A16 on sm70** — a JIT CUDA path for FP4 weights on hardware with no FP4 support, plus the Marlin V100 GPTQ/AWQ repack kernels.
- **TileLang attention for Volta** (`tilelang_fa_v100`) — paged prefill, decode and verify kernels, registered as a first-class attention backend.
- **QSA sparse attention** with a compressed index cache, and its own KV pool (`QSATokenToKVPool`) carrying the compressed-key buffers.
- **GDN linear attention** in TileLang and Triton, tuned for sm70 occupancy.
- **TurboMind sm70 backend** for block-FP8 and FP16 MoE, plus an exact AWQ dequantiser.
- **FP16 / FP8-E5M2 KV cache** on hardware without native FP8. The FP16 path — the production dtype — uses a fast sm70 sparse decode kernel that reads only the selected top-k K/V, so its higher per-token precision costs no decode speed on this sparse-attention model.
- **PLE host offload** — the 51 GB n-gram table lives in host RAM, with the per-request n-gram and short-conv state riding the mamba slot lifecycle.
- **Single-stage custom all-reduce**, because two-stage is pathological on a partial NVLink mesh.
- **fp16 forcing** throughout, since Volta has no bf16 (`SGLANG_SM70_FORCE_FP16`).

## Limitations and known gaps

- **Qwen3.8-Flash-Next is the soaked model.** DeepSeek-V4.1-Flash on this snapshot has run a multi-hour Claude Code session on one conversation (prefix reuse at recorded stops, no crash in that session). It is still one conversation: a second session prefills from zero, and the image does not survive a restart. A long uncached suffix is about 60 tok/s. Leftover HBM after load is a few GiB, and open-ended greedy (temperature 0) can loop. Other architectures may load; several upstream model paths still assume sm80+ kernels.
- **`multimodal_gen` (diffusion / video generation) is not ported.** It carries upstream's code, not this fork's Volta adaptations. The Qwen3.8 vision tower is a different subsystem, and it works. DeepSeek-V4.1 image requests stream the rank-0 tower through GPU GEMMs.
- **Stability was hammered, not soaked.** A ~1-hour sustained load — agentic prompts at np 1/2/4 plus a beyond-spec 32k-token / np 8 phase — ran with no crash and no incorrect output at the current `--mem-fraction-static 0.86`. It did surface one prefill OOM at the previous 0.88 default under the beyond-spec load; the 0.86 retune fixed it (rationale in the serve-script comment). A multi-day soak has not been run.
- **Greedy output is not bit-reproducible across cache states.** A property of the FP16 mamba-hybrid pipeline with a radix cache: the cache replays an approximate GDN (linear-attention) state for a cached prefix, so a prompt's exact tokens can differ a little between a cold and a warm prefix, and prompts sitting on a token decision boundary can vary across runs. Every output is a valid completion.
- **A cold FlashInfer JIT cache costs several minutes** on first launch, and four TP ranks will compile in parallel. Subsequent launches are fast.
- **The dense NVFP4 linear path is unverified.** It matters only if a checkpoint quantises weights outside the MoE experts; Qwen3.8-Flash-Next does not.

## Relationship to upstream

This is a downstream of [haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100), which is itself a fork of [sgl-project/sglang](https://github.com/sgl-project/sglang). The V100 port was cut from upstream around 2026-06-01 and had not been re-synced since; this repository re-lands it onto upstream `main` as of 2026-09-02 (`99b910955`), about 4,250 commits later. Upstream's engine — including the unified radix cache, the hierarchical KV cache and the speculative decoding stack — is used as-is wherever possible; this fork adds the sm70 layer, Qwen3.8-Flash-Next, and an initial DeepSeek-V4.1-Flash serve path on top.

This is not a pure 3-way merge between the two upstream repos. Beyond re-landing the port, the tree carries hand-crafted optimizations and bug fixes, and it is ruggedized, tested, and plug-and-play — it runs as shipped. It is also ongoing: we intend to keep pulling in upstream improvements as well as continuing our own work on top.

Every deviation from upstream carries its reasoning in the commit that made it; `git log` is the record.

Bug reports about the sm70 path belong here. Bug reports about SGLang itself belong upstream.

## Credits

**The Volta port is [haohervchb](https://github.com/haohervchb/sglang-V100)'s work.** Every sm70 kernel in here — the TileLang attention backend, QSA, the GDN linear-attention kernels, NVFP4 on hardware with no FP4 support, the TurboMind sm70 backend, the PLE host offload, the Qwen4-Exp model support — was written there, along with the serving recipe and the tuning that makes it fit in 32 GB. If this is useful to you, that is where the credit belongs. The patched sm70 FlashInfer the build uses is also theirs ([haohervchb/flashinfer](https://github.com/haohervchb/flashinfer)).

This repository's contribution is narrower: re-landing that port onto an SGLang roughly 4,250 commits newer, fixing what the move broke, and adding the DeepSeek-V4.1-Flash path.

Both are derivative works of [SGLang](https://github.com/sgl-project/sglang) (Apache 2.0, Copyright 2023-2024 SGLang Team), which does the hard part.

The Volta build also stands on [marlin_v100](https://github.com/zhinianqin/marlin_v100), [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) (TurboMind sm70), [CUTLASS](https://github.com/NVIDIA/cutlass), [FlashInfer](https://github.com/flashinfer-ai/flashinfer) and [TileLang](https://github.com/tile-ai/tilelang). None are redistributed here — the build fetches them at pinned revisions. Full attribution in [NOTICE](NOTICE).

## License and disclaimer

Apache 2.0, inherited from SGLang — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

This is an **independent community fork**. It is not affiliated with, endorsed by, or supported by the SGLang project, LMSYS, NVIDIA, or the model's authors.

Provided **as is, without warranty or condition of any kind**, per Section 7 of the Apache License. It drives hardware its vendor no longer supports, using kernels written specifically for that purpose; validate it in your own environment before relying on it for anything that matters.

No model weights are distributed here. Any checkpoint you use remains subject to its own license and terms, which you must satisfy independently.

## Contact

Issues and pull requests are the preferred channel. For anything that does not belong in public, `git@jens-david-consulting.com`.
