"""Prefill / decode / concurrency sweep against one serving instance.

Three things are being separated, because on this engine they scale differently
and a single mixed number hides all of it:

  pp   prompt processing. Isolated with an 8-token output so the run is almost
       entirely prefill, and reported as input_len / TTFT rather than the
       harness's input_throughput, which divides by the whole run and so folds
       decode back in.
  tg   token generation. Isolated with a short prompt so TTFT is negligible,
       and read from TPOT (per stream) and output_throughput (aggregate).
  np   concurrent requests, 1..4. The server runs --max-running-requests 4, so
       4 is the ceiling; past that requests queue and the numbers measure the
       queue rather than the engine.

accept_length is carried through every row: this build runs EAGLE MTP, so decode
throughput is a function of how many drafted tokens survive verification, and a
tg number without it cannot be compared against another configuration.

The prefix cache is flushed between configurations. Without that, a repeat at
the same input length scores its predecessor's cache rather than a prefill.

Usage: perf_sweep.py --port 30000 [--out perf-sweep.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request

# Whatever interpreter runs this script; bench_serving is imported from the
# same environment, so no venv path is baked in.
PYTHON = sys.executable
MODEL = "qwen38next-nvfp4"


# (label, input_len, output_len, concurrency, num_prompts)
def build_matrix() -> list[tuple]:
    rows = []
    # pp: prefill across prompt sizes, one stream. Fewer prompts at the long end
    # only because each 131k prefill costs ~35 s.
    for ilen, n in ((512, 6), (2048, 6), (8192, 6), (32768, 4), (131072, 2)):
        rows.append((f"pp/{ilen}", ilen, 8, 1, n))
    # tg: decode across concurrency, prompt short enough that TTFT is noise.
    for np_ in (1, 2, 3, 4):
        rows.append((f"tg/np{np_}", 512, 256, np_, max(8, 4 * np_)))
    # mixed: an agentic-shaped request, prefill and decode both material.
    for np_ in (1, 2, 3, 4):
        rows.append((f"mix/np{np_}", 8192, 512, np_, 4 * np_))
    # long context, the shape the 131.5K+300 gate uses.
    rows.append(("long/131k", 131072, 300, 1, 2))
    return rows


def flush(port: int) -> None:
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{port}/flush_cache", method="POST"
            ),
            timeout=60,
        ).read()
    except Exception as e:
        print(f"  (flush_cache failed: {e})")


def run_one(port: int, label, ilen, olen, conc, nprompts) -> dict | None:
    out_file = f"/tmp/bench_{label.replace('/', '_')}.jsonl"
    open(out_file, "w").close()
    cmd = [
        PYTHON,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang-oai",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        MODEL,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(ilen),
        "--random-output-len",
        str(olen),
        # 1.0 => every prompt is exactly random-input-len, so a row's pp number
        # is not an average over a spread of lengths.
        "--random-range-ratio",
        "1.0",
        "--num-prompts",
        str(nprompts),
        "--max-concurrency",
        str(conc),
        "--warmup-requests",
        "1",
        "--output-file",
        out_file,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5400)
    if proc.returncode != 0:
        print(f"  FAILED rc={proc.returncode}: {proc.stderr.strip()[-300:]}")
        return None
    try:
        rec = json.loads(open(out_file).readline())
    except Exception as e:
        print(f"  no result parsed: {e}")
        return None
    rec["_label"], rec["_wall_s"] = label, time.time() - t0
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11436)
    ap.add_argument("--out", default="perf-sweep.json")
    args = ap.parse_args()

    matrix = build_matrix()
    results = []
    for i, (label, ilen, olen, conc, nprompts) in enumerate(matrix, 1):
        print(
            f"[{i}/{len(matrix)}] {label}: in={ilen} out={olen} np={conc} n={nprompts}",
            flush=True,
        )
        flush(args.port)
        rec = run_one(args.port, label, ilen, olen, conc, nprompts)
        if rec:
            ttft = rec["mean_ttft_ms"] / 1000.0
            pp = ilen / ttft if ttft > 0 else float("nan")
            tg_stream = 1000.0 / rec["mean_tpot_ms"] if rec["mean_tpot_ms"] else 0.0
            print(
                f"    pp={pp:8.1f} tok/s  tg/stream={tg_stream:6.2f} tok/s  "
                f"tg/agg={rec['output_throughput']:7.2f} tok/s  "
                f"accept={rec.get('accept_length')}  ({rec['_wall_s']:.0f}s)",
                flush=True,
            )
            rec["_pp_tok_s"], rec["_tg_stream_tok_s"] = pp, tg_stream
            results.append(rec)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
