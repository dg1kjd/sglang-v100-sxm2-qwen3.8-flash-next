#!/usr/bin/env bash
# V1 launch shape for official deepseek-ai/DeepSeek-V4.1-Flash on 8xV100.
# Mixed MXFP4 experts + MXFP8 dense (packed e4m3+UE8M0 on SM70; GEMV at decode).
# SGLANG_DSV41_MXFP8_W8A16=0 unpacks dense MXFP8 to FP16.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/sglang-v100-venv/bin:$PATH"
export PYTHONPATH="${ROOT}/python${PYTHONPATH:+:$PYTHONPATH}"
export CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 CUDAHOSTCXX=/usr/bin/g++-14
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-14"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

MODEL="${MODEL_PATH:-$HOME/models/DeepSeek-V4.1-Flash}"
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
# This machine: all 8 V100s sit on NUMA node 1; 194×1G hugepages on node 1.
export SGLANG_DSV41_ENGRAM_NUMA_NODE="${SGLANG_DSV41_ENGRAM_NUMA_NODE:-1}"
export SGLANG_DSV41_ENGRAM_NUMA_SPLIT=1
# Private shards: each rank maps 1/TP of Engram. Shared memfd is one physical
# copy but TP RSS mappings of the full table, which the OOM killer sums.
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT="${SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT:-private}"
# Do not posix_fadvise(DONTNEED) the 476 GiB checkpoint; this array writes
# swap at ~80 MB/s once page cache is gone.
export SGLANG_ENABLE_DSV41_ENGRAM_DROP_PAGE_CACHE="${SGLANG_ENABLE_DSV41_ENGRAM_DROP_PAGE_CACHE:-0}"
# WO-11 ablation. 1 = skip Engram residual (tables still mapped). Default 0.
export SGLANG_DSV41_ENGRAM_ZERO="${SGLANG_DSV41_ENGRAM_ZERO:-0}"
# WO-12 overlap. 1 = host gather on a side stream. Default on.
export SGLANG_DSV41_ENGRAM_OVERLAP="${SGLANG_DSV41_ENGRAM_OVERLAP:-1}"
# Coarse HBM knob (GiB/rank of routed MXFP4 on host). Default 10 for greedy
# Tree, 12 when DSPARK=1 (D15-0 slack). Set before this script to override.
# Do not iterate expert counts in engine code.
# 10 GiB/rank spill is required to fit 32 GiB HBM on this 8×V100. LRU
# ensure() is an eager breakable-CUDA-graph node; decode graphs stay on.
export SGLANG_DSV41_EXPERT_SPILL_APPLY="${SGLANG_DSV41_EXPERT_SPILL_APPLY:-1}"
# WO-13 D1: stripe the pinned spill mirror over both sockets (layer i -> node).
export SGLANG_DSV41_EXPERT_SPILL_NUMA_NODES="${SGLANG_DSV41_EXPERT_SPILL_NUMA_NODES:-1,0}"
# WO-13 D2: per-(layer, rank) cold set from recorder dumps
# (scripts/dsv41_cold_set_from_dumps.py). Unset/missing file -> tail placement.
DSV41_COLD_SET_DEFAULT="$HOME/dsv41-v100-logs/cold-set/dsv41_cold_set.pt"
if [[ -z "${SGLANG_DSV41_EXPERT_SPILL_COLD_SET:-}" && -f "${DSV41_COLD_SET_DEFAULT}" ]]; then
  export SGLANG_DSV41_EXPERT_SPILL_COLD_SET="${DSV41_COLD_SET_DEFAULT}"
fi
# SM70 wo_a is dense MXFP8 (packed e4m3); DeepGEMM fp8_einsum is SM90+.
export SGLANG_OPT_FP8_WO_A_GEMM="${SGLANG_OPT_FP8_WO_A_GEMM:-0}"

# SM70 keeps dense MXFP8 packed (about half the FP16 unpack footprint).
# SGLANG_DSV41_ATTN_GLUE=0 restores pack + index_copy + KV gather.
# SGLANG_DSV41_HOST_GEMV=1 enables host MXFP4 GEMV for spilled decode hits
# (default off: packed CPU kernel still slower than D4-G UVA landing).
# WO-M 0.88 assumed packed MXFP8; unpack-to-FP16 left ~0.58 GiB free.
# 0.99 keeps ~260 MiB for the 890 B/tok pool. Decode BCG is bs=1 (np=1).
# V4 default max_running_requests=256 allocates req_to_token
# [257, context_len] int32. v1 is np=1. SM70 CSA2 keeps SWA on rings,
# so the paged SWA pool is a fixed floor (chunked-prefill admission +
# sticky/running windows) and 256k fits next to DSpark draft + landing-36.
# A window-only 1024-token floor livelocked 8k prefill (relaunch148).
# 144 OOM'd 8k when that pool still scaled with context (2394 B/tok →
# 135k profile, then CSA2@256k). Override with SGLANG_DSV41_CONTEXT_LEN.
# Do not call /health (drops the sticky pin).
# Pre-warm NCCL while the card is empty.
# Decode is ~100 small f16 all-reduces/token. RING_LL on this hybrid mesh was
# ~73% of GPU time (relaunch55). Tree for AllReduce; do not pin PROTO=LL
# (that forced LL on 20 MiB prefill ARs and slowed 8k TTFT). WO-14: in-graph
# pair+quad custom-AR is capturable but RankSignals-bound (~67 ms/tok) while
# spill_copy is rank-divergent. Default stays 8-rank Tree. HIER_AR_CA stays 0.
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-2097152}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-1}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-4}"
export NCCL_ALGO="${NCCL_ALGO:-allreduce:tree}"
export SGLANG_DSV41_HIER_AR="${SGLANG_DSV41_HIER_AR:-0}"
export SGLANG_DSV41_HIER_AR_CA="${SGLANG_DSV41_HIER_AR_CA:-0}"
# WO-15 DSpark. Default on: best measured TG on this 8×V100 box (coding
# ~8 tok/s, α~5.9). The ≥15 tok/s line is not met. DSPARK=1 sets landing
# 36, spill 13 (12 left 8 MiB short of Engram's 300 MiB 8k unpack),
# Markov BF16 off, and the speculative CLI flags. Keep
# max-running-requests at 1 (the DSpark hook would otherwise become 48).
# Set SGLANG_DSV41_DSPARK=0 to restore greedy Tree.
export SGLANG_DSV41_DSPARK="${SGLANG_DSV41_DSPARK:-1}"
export SGLANG_DSPARK_OPT_MARKOV_W2_BF16="${SGLANG_DSPARK_OPT_MARKOV_W2_BF16:-0}"
export SGLANG_DSPARK_FAST_KERNEL="${SGLANG_DSPARK_FAST_KERNEL:-1}"
SPEC_FLAGS=()
if [[ "${SGLANG_DSV41_DSPARK}" == "1" ]]; then
  export SGLANG_DSV41_SPILL_LANDING="${SGLANG_DSV41_SPILL_LANDING:-36}"
  export SGLANG_DSV41_EXPERT_SPILL_GB="${SGLANG_DSV41_EXPERT_SPILL_GB:-13}"
  # 256k is the ship window (page-aligned). CSA2 kv_rows scale with this
  # value; 512k has not been shown to leave 300 MiB for Engram unpack.
  export SGLANG_DSV41_CONTEXT_LEN="${SGLANG_DSV41_CONTEXT_LEN:-262144}"
  # WO-15 Step 2: T=6 TARGET_VERIFY is packed CSA2 (no positions[0].item()).
  # Capture CSA2+mHC in the verify graph. Rollback 1 if capture hits a host sync.
  export SGLANG_DSV41_EAGER_CSA2_HC="${SGLANG_DSV41_EAGER_CSA2_HC:-0}"
  # D15-0: 0.99 spends slack on KV; T=6 verify capture then OOMs unpacking
  # Engram MXFP8 wkv (~300 MiB, relaunch115). 0.88 is the floor that still
  # allocates a KV pool after draft weights (0.87 raises).
  export SGLANG_DSV41_MEM_FRACTION="${SGLANG_DSV41_MEM_FRACTION:-0.88}"
  SPEC_FLAGS+=(--speculative-algorithm DSPARK --speculative-draft-model-path "${MODEL}")
else
  export SGLANG_DSV41_SPILL_LANDING="${SGLANG_DSV41_SPILL_LANDING:-6}"
  export SGLANG_DSV41_EXPERT_SPILL_GB="${SGLANG_DSV41_EXPERT_SPILL_GB:-10}"
  export SGLANG_DSV41_CONTEXT_LEN="${SGLANG_DSV41_CONTEXT_LEN:-262144}"
fi
export SGLANG_DSV41_CONTEXT_LEN="${SGLANG_DSV41_CONTEXT_LEN:-262144}"
# D5: one plain decode graph when D4-G landing is on. Prefill stays eager.
# Breakable remains if MoE/CSA2/Engram rollbacks are set, landing is off,
# or SGLANG_DSV41_BREAKABLE_DECODE=1. APPLY=0 can capture bs>1.
export SGLANG_DSV41_SPILL_LANDING="${SGLANG_DSV41_SPILL_LANDING:-6}"
GRAPH_FLAGS=(--disable-prefill-cuda-graph)
_d5_breakable=0
if [[ "${SGLANG_DSV41_EXPERT_SPILL_APPLY}" == "0" ]]; then
  GRAPH_FLAGS+=(--cuda-graph-max-bs-decode 4)
else
  if [[ "${SGLANG_DSV41_BREAKABLE_DECODE:-0}" == "1" \
     || "${SGLANG_DSV41_EAGER_MOE_SPILL:-0}" == "1" \
     || "${SGLANG_DSV41_EAGER_CSA2_HC:-0}" == "1" \
     || "${SGLANG_DSV41_EAGER_ENGRAM:-0}" == "1" \
     || "${SGLANG_DSV41_SPILL_LANDING}" == "0" ]]; then
    _d5_breakable=1
  fi
  if [[ "${_d5_breakable}" == "1" ]]; then
    GRAPH_FLAGS+=(--cuda-graph-backend-decode breakable --cuda-graph-max-bs-decode 1)
  else
    GRAPH_FLAGS+=(--cuda-graph-max-bs-decode 1)
  fi
fi
# SM70 CSA2 packed SWA/KV lives on the attention backend, not in the radix
# tree. A radix prefix hit skips hidden states and desyncs ratio-2 pending
# (turn-3 crash: pos 511 layer 2). Keep radix off. One-slot token-id
# continuation (exact last finished sequence only) pins KV across HTTP
# turns of one full-history chat. Partial prefix /health drops the pin.
GRAPH_FLAGS+=(--disable-radix-cache)
export SGLANG_DSV41_STICKY_LAST_SEQ="${SGLANG_DSV41_STICKY_LAST_SEQ:-1}"

# Optional Nsight wrap. Capture starts at cudaProfilerStart (HTTP /start_profile
# activities=["CUDA_PROFILER"]). Stacks/sample off: they segfaulted on this box.
NSYS_WRAP=()
if [[ -n "${SGLANG_DSV41_NSYS_OUT:-}" ]]; then
  NSYS_BIN="${NSYS_BIN:-$(command -v nsys || true)}"
  NSYS_BIN="${NSYS_BIN:-/usr/local/cuda-12.9/bin/nsys}"
  NSYS_WRAP=(
    "${NSYS_BIN}" profile
    --force-overwrite=true
    --trace=cuda,nvtx
    --sample=none
    --cpuctxsw=none
    --backtrace=none
    --cudabacktrace=none
    --cuda-event-trace=false
    --stats=false
    --cuda-graph-trace=node
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --kill=none
    -o "${SGLANG_DSV41_NSYS_OUT}"
  )
fi

exec "${NSYS_WRAP[@]}" python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --tp 8 --ep-size 8 \
  --dtype float16 \
  --moe-runner-backend marlin \
  --attention-backend dsv4 \
  --context-length "${SGLANG_DSV41_CONTEXT_LEN}" \
  --chunked-prefill-size 2048 \
  --mem-fraction-static "${SGLANG_DSV41_MEM_FRACTION:-0.99}" \
  --max-running-requests 1 \
  --max-total-tokens "${SGLANG_DSV41_CONTEXT_LEN}" \
  --max-prefill-tokens "${SGLANG_DSV41_CONTEXT_LEN}" \
  --pre-warm-nccl \
  "${GRAPH_FLAGS[@]}" \
  --language-model-only \
  --reasoning-parser deepseek-v41 \
  --tool-call-parser deepseekv41 \
  --trust-remote-code \
  --disable-custom-all-reduce \
  "${SPEC_FLAGS[@]}" \
  --host 0.0.0.0 \
  --port "${PORT:-30000}" \
  "$@"
