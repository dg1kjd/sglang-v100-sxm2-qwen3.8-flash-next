"""Verify the published tree is the development tree minus exactly what we sanction.

`publish` is a separate lineage from `main`, not a descendant of it (the public
repo keeps a short curated history rooted at a single vendored-upstream commit
rather than the ~6k commits of the full merge). Git therefore cannot express
"publish differs from main only in the ways we allow" as a branch relationship,
so this asserts it directly instead:

  1. every path that differs between the two trees sits under an allowed prefix,
     and publish adds nothing main does not have;
  2. the published tree carries no host paths, internal addresses or credentials.

Check 2 is the one a parent-child link could never give us: an ancestry edge
says the trees differ somehow, it does not say the difference is safe to
publish. Run this before every push to the public remote.

Patterns are deliberately specific to this deployment rather than generic. A
broad sweep is worse than useless here -- `192\\.168\\.` matches 10 upstream doc
files and a bare `10\\.x\\.x\\.x` matches 47 files of version strings, so a
scanner built on those would be ignored within a week. What must be zero is our
own infrastructure and anything shaped like a real credential.

Usage:
    python3 scripts/check_publish_tree.py [--dev main] [--publish publish]
Exit status is 0 when the published tree is clean, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

# Paths allowed to differ. Upstream's CI is stripped from the public tree: those
# workflows target sgl-project's runners and secrets, so on a fork GitHub would
# schedule 109 jobs that cannot pass.
ALLOWED_DIFF_PREFIXES = (".github/workflows/",)

# Assembled from fragments so this file does not match its own patterns. The
# obvious alternative -- excluding this path from the scan -- would put a blind
# spot in the one file that defines what a leak looks like.
_SUBNET = r"192\.168\." + r"225\."
_HOME = "/home/" + "nvi" + "dia"
_HOST = r"\b" + "ins" + r"pur\b"
_SPILL = r"/home/[a-z0-9_-]+/hicache" + "_storage"

# Deployment-specific strings that must never reach the public tree.
PRIVATE_MARKERS = {
    "internal address": _SUBNET,
    "operator home path": _HOME,
    "deployment hostname": _HOST,
    "hicache spill path": _SPILL,
}

# Credential shapes. These carry their own entropy, so a hit is a hit.
CREDENTIAL_PATTERNS = {
    "GitHub token": r"gh[pousr]_[A-Za-z0-9]{20,}",
    "HuggingFace token": r"\bhf_[A-Za-z0-9]{30,}",
    "OpenAI-style key": r"\bsk-[A-Za-z0-9]{20,}",
    "AWS access key": r"\bAKIA[0-9A-Z]{16}\b",
    "Slack token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
}

# Our own domains. The public contact address is deliberate; anything else on
# these domains is a leak.
OUR_DOMAINS = r"jens-david-consulting\.com|" + "anthr" + "2026"
SANCTIONED_ADDRESSES = {"git@jens-david-consulting" + ".com"}

# A PEM header alone is not a key -- upstream's mTLS test asserts on the header
# string (sgl-model-gateway/tests/security/mtls_test.rs). Require a base64 body
# line too, which only an embedded key actually has. POSIX ERE: `git grep -E`
# rejects Perl constructs like (?:...), and an unsupported pattern would make
# this scan silently match nothing.
PRIVATE_KEY_HEADER = r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
PRIVATE_KEY_BODY = re.compile(r"^[A-Za-z0-9+/=]{40,}$", re.MULTILINE)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout


def grep_ref(ref: str, pattern: str) -> list[str]:
    """Lines in `ref` matching `pattern`, as '<ref>:<path>:<lineno>:<text>'."""
    # -e is required, not stylistic: the PEM pattern starts with "-----" and git
    # would otherwise parse it as an option and print its usage.
    proc = subprocess.run(
        ["git", "grep", "-I", "-n", "-E", "-e", pattern, ref, "--", "."],
        capture_output=True,
        text=True,
        check=False,
    )
    # 0 = matched, 1 = no match. Anything else is a broken pattern or a bad ref,
    # which must abort rather than read as "clean" -- a scanner that no-ops on a
    # malformed regex is worse than no scanner.
    if proc.returncode > 1:
        raise RuntimeError(
            f"git grep failed for pattern {pattern!r}: {proc.stderr.strip()}"
        )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def check_tree_divergence(dev: str, publish: str) -> list[str]:
    # name-status, so an unexpected path is reported once, with whether publish
    # added, modified or dropped it.
    status = {"A": "added in publish", "M": "modified in publish", "D": "dropped"}
    failures = []
    for line in git("diff", "--name-status", dev, publish).splitlines():
        if not line.strip():
            continue
        code, _, path = line.partition("\t")
        path = path.strip()
        if not path or path.startswith(ALLOWED_DIFF_PREFIXES):
            continue
        failures.append(
            f"unsanctioned divergence [{status.get(code[:1], code)}]: {path}"
        )
    return failures


def check_secrets(publish: str) -> list[str]:
    failures = []

    for label, pattern in {**PRIVATE_MARKERS, **CREDENTIAL_PATTERNS}.items():
        for hit in grep_ref(publish, pattern):
            failures.append(f"{label}: {hit}")

    for hit in grep_ref(publish, OUR_DOMAINS):
        if not any(addr in hit for addr in SANCTIONED_ADDRESSES):
            failures.append(f"non-public address on our domain: {hit}")

    for hit in grep_ref(publish, PRIVATE_KEY_HEADER):
        path = (
            hit.split(":", 2)[1] if hit.startswith(publish + ":") else hit.split(":")[0]
        )
        blob = git("show", f"{publish}:{path}")
        if PRIVATE_KEY_BODY.search(blob):
            failures.append(f"embedded private key: {path}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="main")
    ap.add_argument("--publish", default="publish")
    args = ap.parse_args()

    for ref in (args.dev, args.publish):
        if not git("rev-parse", "--verify", "--quiet", ref).strip():
            print(f"FAIL: no such ref: {ref}")
            return 1

    divergence = check_tree_divergence(args.dev, args.publish)
    secrets = check_secrets(args.publish)

    allowed = ", ".join(ALLOWED_DIFF_PREFIXES)
    print(f"tree divergence {args.dev} -> {args.publish} (allowed: {allowed})")
    print(f"  {'FAIL' if divergence else 'ok'}: {len(divergence)} unsanctioned path(s)")
    print(f"secret scan of {args.publish}")
    print(f"  {'FAIL' if secrets else 'ok'}: {len(secrets)} finding(s)")

    for line in divergence + secrets:
        print(f"    {line}")

    if divergence or secrets:
        print("\nPUBLISH BLOCKED")
        return 1
    print("\nclean; safe to push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
