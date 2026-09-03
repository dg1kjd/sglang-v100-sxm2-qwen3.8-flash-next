# Syncing with upstream SGLang

Measured 2026-09-03, after the re-land was validated.

| | |
|---|---|
| validated pin | `99b910955` (2026-09-02 17:51 +0800) |
| upstream now | `27b7a2dc3` (2026-09-03 17:37 +0800) |
| new commits | **53** |
| files changed | **1,585** |
| overlap with the 167 files we hand-resolved | **83** |

## Recommendation: do not merge these 53 yet

Not because they are unwelcome, but because of *one* of them.

`28262c20d [CI][RFC] Replace black-jupyter with ruff-format (#37210)` is a
**tree-wide reformat: 1,411 files, +7,767 / -8,177**. That single commit is why
53 commits touch 1,585 files, and why 83 of our 167 hand-resolved files are in
the overlap. Merging it as-is would re-conflict most of the re-land's careful
work against changes that carry no meaning.

Upstream shipped `57c26a84e .git-blame-ignore-revs` alongside it, which is the
tell: they expect tooling, not humans, to absorb it.

## How to absorb it cleanly, when you do sync

Run the formatter on our side *first*, so the reformat is a no-op by the time
the merge sees it and only semantic conflicts remain:

1. Adopt upstream's formatter config (the `ruff-format` settings from #37210).
2. Run `ruff format` over the fork-owned files on a branch off `reland`, and
   commit that alone -- a pure-formatting commit, verifiable with
   `ruff format --check` and by confirming the AST is unchanged.
3. *Then* merge upstream. `rerere` still holds the 155 resolutions from this
   re-land, so the semantic conflicts that remain should largely replay.
4. Add our formatting commit to `.git-blame-ignore-revs` too.

The `mechanical-refactor-verify` skill in `.claude/skills/` is written for
exactly this shape: prove the transform is reproducible rather than eyeballing
a 1,411-file diff.

## What is actually worth having from these 53

Most of the 53 are irrelevant to V100 (AMD/ROCm, XPU, CPU base images, Kimi K3,
K2 Horizon, docs, CI). The substantive ones that touch our risk areas:

- `cf3173aeb` [Perf] Walk the radix tree by offset instead of re-slicing token
  storage — radix-cache hot path, plausibly a real win for long agentic prefixes.
- `5ddca6819`, `5a1275a51`, `d9848b9ec`, `18d5ffb42` — unified-SWA / unified
  read-table fixes. We do not run `--enable-unified-memory`, so these are
  latent-value only.
- `87d60a222` Improve CUDA graph and speculative execution output handling —
  worth reading against our MTP path.
- `4229088a4` feat(kernels): generalize persistent CuTe JIT cache — could cut
  the SM70 JIT warm-up cost.
- `3fa6b8650` [Spec] Publish the final multi-layer EAGLE shared-read event.

None of these fixes a problem we currently have. The engine is validated on the
current pin and beats its performance baseline; a sync is an improvement
exercise, not a repair.

## Cost check for a later sync

Upstream moves ~50 commits/day, so drift is ~1 day per 50. The re-land's own
mechanics (the path-normalisation step, the synthetic graft, the
import sweep) are all re-runnable, and this document documents the
procedure. A sync at this size should be hours, not the multi-day effort the
original 4,250-commit gap required -- provided the reformat is handled as above
rather than merged head-on.
