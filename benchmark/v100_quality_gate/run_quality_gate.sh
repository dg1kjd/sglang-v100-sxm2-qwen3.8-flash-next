#!/usr/bin/env bash
# Intelligence / correctness gate for the re-landed V100 engine.
#
# Each benchmark below is pinned to the protocol of the number we compare it
# against, because the comparison is only meaningful at the same settings:
#
#   gsm8k   -> the checkpoint's own gsm8k_metrics.json (score 0.9727), which
#              was produced by sgl-eval at t0.6 / top-p 0.95 / max 8192 /
#              seed 0 / 1319 examples. Unquantised band from the model card:
#              97.12-97.50 (BF16, earlier revision of the same line).
#   aime26  -> the checkpoint's own aime26_metrics.json (pass@1 0.9875) at
#              t1.0 / max 130k / thinking / 8 repeats. Expensive: 4.9M
#              completion tokens, budget most of a day on this box.
#   gpqa    -> Qwen's published number for the *unquantised* base model,
#              GPQA Diamond 91.7. Qwen's thinking-mode recipe is
#              t1.0 / top-p 0.95 / top_k 20 / min_p 0; sgl-eval does not
#              expose top_k, so this reproduces every knob except that one --
#              the same gap the checkpoint's own aime26 baseline has.
#   ruler2  -> no published reference; run it for the shape of the
#              accuracy-vs-context curve, which is where the SM70 attention
#              and fp8 KV paths would show up.
#
# Usage:  run_quality_gate.sh <gsm8k|gpqa|aime26|ruler2> [extra sgl-eval args]
set -euo pipefail

BENCH="${1:?usage: run_quality_gate.sh <gsm8k|gpqa|aime26|ruler2> [args...]}"
shift || true

CLIENT="${SGL_EVAL_BIN:-sgl-eval}"
BASE_URL="${GATE_BASE_URL:-http://127.0.0.1:11400/v1}"   # the lb proxy by default
MODEL="${GATE_MODEL:-qwen38next-nvfp4}"
OUT="${GATE_OUT_DIR:-$(dirname "$0")/results}"

case "$BENCH" in
  gsm8k)
    ARGS=(--num-examples 1319 --num-threads 16 --n-repeats 1
          --max-tokens 8192 --temperature 0.6 --top-p 0.95 --seed 0) ;;
  gpqa)
    # 48k output budget, not the 32k a non-reasoning run would need: this model
    # reasons long (its own aime26 baseline was run at max_tokens 130000), and a
    # budget that truncates mid-thought scores as a wrong answer, which would
    # read as a quality regression that is really a harness setting. Only the
    # few long responses pay for the headroom. Watch truncated_rate: above a few
    # percent the score is measuring the budget, not the model.
    # --from-dataset: sgl-eval's own loader pulls the gated Idavidrein/gpqa off
    # the Hub and dies with DatasetNotFoundError here (no HF token). The jsonl
    # is the same 198 Diamond rows rebuilt from the ungated simple-evals CSV
    # through the vendored formatter -- see build_gpqa_diamond.py.
    ARGS=(--num-threads 16 --n-repeats 1 --thinking
          --max-tokens 49152 --temperature 1.0 --top-p 0.95
          --from-dataset "$(dirname "$0")/data/gpqa_diamond.jsonl") ;;
  aime26)
    ARGS=(--num-examples 30 --num-threads 8 --n-repeats 8 --thinking
          --max-tokens 130000 --temperature 1.0 --top-p 0.95 --seed 0) ;;
  ruler2)
    ARGS=(--num-threads 4 --max-tokens 8192 --temperature 0.0) ;;
  *) echo "unknown benchmark: $BENCH" >&2; exit 1 ;;
esac

mkdir -p "$OUT"
export OPENAI_API_KEY=EMPTY
set -x
exec "$CLIENT" run "$BENCH" \
  --base-url "$BASE_URL" --model "$MODEL" --out-dir "$OUT" \
  "${ARGS[@]}" "$@"
