"""Does batching change what the engine generates?

Everything in the quality gate so far ran saturated (4 concurrent per GPU set),
so a defect that only appears at batch size > 1 would have been invisible: the
scores would just be a bit lower and we would have blamed the quantisation.
This isolates batch size as the only variable.

Accuracy is the wrong instrument for that -- at n=100 its sem is ~4 pp, wide
enough to swallow a real bug. Greedy token equality is far sharper: with
temperature 0 the same prompt must produce the same tokens regardless of what
else shares the batch, so any divergence is signal rather than sampling noise.

Two passes over one prompt set against ONE instance (never the proxy -- the
whole point is controlling which batch a request lands in):

  serial   : one request at a time            -> decode batch size 1
  batched  : `--concurrency` at once          -> decode batch size up to 4,
                                                 which dispatches through the
                                                 bs=2 / bs=4 captured CUDA
                                                 graphs instead of bs=1

Then compare per prompt and report where they first differ.

Interpreting the result: bit-level batch invariance is NOT guaranteed -- reduction
order in a batched GEMM legitimately differs from the unbatched one, so a small
number of late divergences is expected and benign. What is not benign is a high
divergence rate, or divergences that start in the first handful of tokens, or
any divergence in the prompt-independent prefix. Those mean state is leaking
across concurrent sequences (the mamba pool under --mamba-radix-cache-strategy
extra_buffer and EAGLE verify are the candidates here), not that floating point
is associative-ish.

Usage:
  batch_invariance.py --base-url http://SERVER_HOST:30000 [--n 40]
                      [--max-tokens 256] [--concurrency 4]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request

# Deliberately mundane prompts with a long, deterministic continuation: the test
# needs many greedy tokens per request to have something to diverge on, not
# interesting content. Reasoning is left off so the comparison is over visible
# output rather than a thinking trace the parser may withhold.
PROMPTS = [
    "Count from 1 to 60, separated by commas, then stop.",
    "List the first 40 prime numbers, separated by commas.",
    "Recite the alphabet, then the alphabet backwards, then stop.",
    "Write the multiplication table for 7, from 7x1 to 7x25, one per line.",
    "Name the planets in order from the Sun, with one sentence each.",
    "List the days of the week and the months of the year, then stop.",
    "Write the squares of 1 through 30, one per line, as 'n: n^2'.",
    "Explain in exactly ten numbered steps how to make a cup of tea.",
    "List the 26 letters of the alphabet, each with a common word starting with it.",
    "Write the Fibonacci sequence up to the 30th term, comma separated.",
]


def generate(base_url: str, prompt: str, max_tokens: int) -> dict:
    body = {
        "model": "qwen38next-nvfp4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        # enable_thinking, NOT thinking: this checkpoint's chat template reads
        # `enable_thinking` and silently ignores anything else, so the first
        # version of this script left reasoning ON while believing it off. That
        # invalidated the comparison -- the model spent the whole budget
        # reasoning, and whether the answer fit at all became the thing being
        # measured.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    msg = out["choices"][0]["message"]
    # Compare the whole emitted stream. Reasoning is where most tokens go when
    # it is on at all, so diffing `content` alone would miss a divergence that
    # happens inside the thinking block and then converges.
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    return {
        "text": text,
        "finish": out["choices"][0]["finish_reason"],
        "completion_tokens": (out.get("usage") or {}).get("completion_tokens"),
        "latency": time.time() - t0,
    }


def first_divergence(a: str, b: str) -> int:
    """Character index where the two outputs first differ, or -1 if identical."""
    if a == b:
        return -1
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="ONE instance, not the lb proxy")
    ap.add_argument("--n", type=int, default=40, help="total requests per pass")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.n)]

    print(f"pass 1/2: serial (batch size 1), {args.n} requests", flush=True)
    t0 = time.time()
    serial = [generate(args.base_url, p, args.max_tokens) for p in prompts]
    serial_wall = time.time() - t0

    print(
        f"pass 2/2: batched (concurrency {args.concurrency}), {args.n} requests",
        flush=True,
    )
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        batched = list(
            ex.map(lambda p: generate(args.base_url, p, args.max_tokens), prompts)
        )
    batched_wall = time.time() - t0

    # A response that ran out of budget was cut mid-stream, so comparing it
    # measures where the cap fell rather than what the engine computed. Count
    # those separately instead of scoring them as divergences.
    truncated = [
        i
        for i, (s, b) in enumerate(zip(serial, batched))
        if s["finish"] == "length" or b["finish"] == "length"
    ]
    divergences = []
    for i, (s, b) in enumerate(zip(serial, batched)):
        if i in truncated:
            continue
        d = first_divergence(s["text"], b["text"])
        if d != -1:
            divergences.append((i, d, len(s["text"]), len(b["text"])))

    comparable = args.n - len(truncated)
    identical = comparable - len(divergences)
    print("\n" + "=" * 60)
    if truncated:
        print(
            f"EXCLUDED          : {len(truncated)} hit max_tokens "
            f"(raise --max-tokens; a cut stream is not comparable)"
        )
    print(
        f"identical outputs : {identical}/{comparable} ({identical / max(comparable, 1):.1%})"
    )
    if divergences:
        idx = [d for _, d, _, _ in divergences]
        print(f"divergent         : {len(divergences)}")
        print(
            f"first-diff char   : min={min(idx)} median={statistics.median(idx):.0f} max={max(idx)}"
        )
        print("  worst offenders (request, first-diff char, len serial, len batched):")
        for row in sorted(divergences, key=lambda r: r[1])[:5]:
            print(f"    {row}")
    print(f"serial wall  : {serial_wall:.1f}s")
    print(
        f"batched wall : {batched_wall:.1f}s  (speedup {serial_wall / batched_wall:.2f}x)"
    )
    print("=" * 60)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "n": args.n,
                    "concurrency": args.concurrency,
                    "max_tokens": args.max_tokens,
                    "identical": identical,
                    "divergences": divergences,
                    "serial_wall_s": serial_wall,
                    "batched_wall_s": batched_wall,
                },
                f,
                indent=1,
            )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
