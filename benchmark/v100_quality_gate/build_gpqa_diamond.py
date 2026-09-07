"""Build sgl-eval's GPQA Diamond jsonl without the gated HuggingFace dataset.

sgl-eval's gpqa loader calls NeMo-Skills' prepare.py, which does
`load_dataset("Idavidrein/gpqa", "gpqa_diamond")` -- gated on the Hub, so it
fails with DatasetNotFoundError on a box with no HF token. The same 198 Diamond
rows are published ungated by OpenAI's simple-evals as a CSV with the identical
column names, so we read that and hand the rows to the *vendored* formatter.

Importing `format_entry` rather than reimplementing it is the point: the
question/choice rendering, the `preprocess` text fixups, the A-D lettering and
the choice shuffle all stay byte-identical to the gated path, so the score
remains comparable to numbers produced from the HF dataset.

The one thing that does not carry over is row order (the CSV's order need not
match the Hub's). Each row still draws its own uniform shuffle from the
seed-42 stream, so which distractor lands on which letter differs from a
gated run; the choice distribution -- and therefore accuracy in expectation --
does not.

Usage: build_gpqa_diamond.py <input.csv> <output.jsonl>
"""

import csv
import json
import random
import sys

from sgl_eval._vendored.nemo_skills.dataset.gpqa.prepare import format_entry

RANDOM_SEED = 42  # prepare.py's save_kwargs in sgl_eval/evals/_registry.py


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]

    with open(src, newline="", encoding="utf-8") as fin:
        rows = list(csv.DictReader(fin))

    random.seed(RANDOM_SEED)
    with open(dst, "w", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            entry = format_entry(row)
            entry["id"] = i
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    letters = {}
    for line in open(dst, encoding="utf-8"):
        a = json.loads(line)["expected_answer"]
        letters[a] = letters.get(a, 0) + 1
    print(f"wrote {len(rows)} rows -> {dst}")
    print(f"answer-letter distribution: {dict(sorted(letters.items()))}")


if __name__ == "__main__":
    main()
