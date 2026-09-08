# Syncing with upstream SGLang

The fork tracks two upstreams. Check both.

| | |
|---|---|
| `sgl-project/sglang` | the engine; moves ~70 commits/day |
| `haohervchb/sglang-V100` | the Volta port this fork re-lands; last moved `72ef2f5f3`, 2026-09-01, and is fully contained here |

## Measured: syncing 93 commits (2026-09-08)

Third sync, `31d28a296` -> `ccfa120da`, one day of drift.

| | |
|---|---|
| new upstream commits | 93 |
| files upstream touched | 1,492 |
| our surface against upstream | 230 |
| raw file overlap | 23 |
| **conflicts `git merge` produced** | **2** |
| new import breakage (G1 sweep) | 1, characterised, unreachable on sm70 |
| AOT kernel rebuild required | no (upstream touched only `csrc/cpu/`) |
| dependency pin changes | none |
| cold 131.5K+300, best of 3 | 37.16 s vs 37.39 s before the merge |

Both conflicts came from one thing: the five-commit **`[Config]` Round 6.x
stack** (#38046 -> #38113), which finished moving configuration off the
`ServerArgs` record and into the runtime context's namespace bags.

- `flashinfer_backend.py` — `server_args.enable_mis` became
  `get_exec().features.enable_mis`. Keep both sides; ours added `self.device`.
- `server_args.py` — every `*_CHOICES` list moved to a new
  `arg_groups/choices.py` and the file shrank from ~4,460 lines to ~1,000. Take
  upstream wholesale and re-register `tilelang_fa_v100`, `flash_attn_v100` and
  the `tilelang` linear-attn backend in the new file. `server_args` re-exports
  the lists, so out-of-tree importers are unaffected.

### A sweep of upstream's own callers is the thing to scan for

Round 6.4 converted sixty-odd files and dropped nine now-empty parameters at
every call site. That is the same shape as the two failures the previous sync's
static gates missed, and the reason is structural: **`merge-ort` only looks at
files both sides changed**, so a tree-wide rename lands everywhere except our
67 fork-only files, silently.

Three scans generalise it. Each diffs the old pin against the new tip, then
checks our surface — `git diff --name-only sglang_live` — for the leftovers:

1. **Signature drift.** Parse every changed file at both refs, diff each
   function's parameter set, then walk our call sites for a removed keyword.
   *(20 functions lost a parameter; 0 of our call sites passed one.)*
2. **Removed record fields.** Diff the `ServerArgs` declarations at both refs.
   *(4 removed, all CP-v1 deprecation; 0 referenced here.)*
3. **Attribute drift.** Collect every attribute our files access, subtract the
   merged tree's whole name vocabulary, and keep what existed at the old pin.
   This is the one that catches a rename like
   `num_token_non_padded_cpu` -> `global_num_token_non_padded_cpu`. *(0.)*

Scan 3 has a false-positive class worth knowing: Round 6.5 turned the parallel
quotients (`attn_tp_size` and five siblings, `is_ep_joiner`,
`is_startup_weight_load_overlap`, `enable_mamba_extra_buffer{,_lazy}`) from
`def` properties into `Derived(...)` declarations. A scan that only recognises
`def` and `class` reports them as deleted. They resolve through the bags
unchanged.

None of the three catches the *second* failure class from last sync — a
predicate getting narrower, with nothing renamed or removed. Only step 7 does.

## Measured: syncing 214 commits (2026-09-07)

Second sync since the re-land, `99b910955` -> `31d28a296`. This is the
reference data point for what a sync of this size actually costs.

| | |
|---|---|
| new upstream commits | 214 |
| files upstream touched | 2,410 |
| files we had touched | 220 |
| raw file overlap | 59 |
| overlap after excluding the tree-wide reformat | **14** |
| **conflicts `git merge` actually produced** | **10** |
| conflict hunks in total | 12 |
| new import breakage (G1 sweep) | **0** — byte-identical to pre-merge |
| AOT kernel rebuild required | no (upstream touched only HIP/ROCm and Python wrappers) |
| dependency pin changes | none affecting `requirements.txt` |

Ten conflicts, all sitting exactly where the sm70 layer meets upstream. Nine
were one hunk each and additive. That is the shape to expect: the fork's
surface area against upstream is small and stable, so sync cost scales with
*how many upstream commits land on that surface*, not with upstream's total
churn.

### The reformat is not the problem it looks like

`28262c20d [CI][RFC] Replace black-jupyter with ruff-format (#37210)` reformats
1,411 files, and it is why the raw overlap is 59 rather than 14. An earlier
revision of this document predicted it would re-conflict most of the re-land's
hand-resolved files and prescribed running the formatter on our side first, as
its own commit, before merging.

**That was wrong, and the measurement above is the correction.** `merge-ort`
absorbed the reformat on its own: of the 45 overlap files that exist only
because of it, *zero* produced a conflict. Reformatting touches whole-file
whitespace and line wrapping, but our changes sit in different hunks, so the
three-way merge resolves them independently. Exactly one conflict
(`marlin_utils.py`, an import line) was formatting-adjacent, and it was a
two-line resolution.

So: **merge head-on.** Do not spend a pre-formatting pass on it. The one real
consequence is cosmetic -- lines the merge takes from our side are not
ruff-formatted, so `ruff format` reports a diff afterwards. Run it as a
follow-up commit if you care, and add that commit to `.git-blame-ignore-revs`.

## Procedure

```bash
git fetch https://github.com/sgl-project/sglang.git main
git branch -f sglang_new FETCH_HEAD

# Size it before committing to it.
git merge-tree --write-tree reland sglang_new | grep '^CONFLICT'

git checkout -b sync-<date> reland
git merge sglang_new
```

Then, in order:

1. **Resolve.** `rerere` holds the re-land's 155 resolutions and replays what it
   can. Read every remaining hunk against the sm70 seams -- upstream adding a
   parallel mechanism for the same problem (a native-FP8 path next to our
   software dequant) is the common case, and the answer is usually "keep both,
   ours first", not "pick a side".
2. **Scan for markers**, matching 7 *and* 8 characters -- rename/rename
   conflicts write `<<<<<<<<`:
   `grep -rlE '^<{7,8}[ A-Za-z]' --include='*.py' python/ test/`
3. **Byte-compile:** `python -m compileall -q -j 8 python/ test/`
4. **Run the import sweep** and diff it against the same sweep on `reland`. A
   sync is clean when the two outputs are *identical*, not when the sweep is
   empty -- there is known pre-existing residue (dead benchmark files under the
   old `jit_kernel/` path).
5. **Check for deleted symbols.** Upstream dedup commits are the real hazard,
   not conflicts. Extract what a refactor removed and grep the merged tree for
   survivors:
   `git show <sha> -- <files> | grep -E '^-(def |class )'`
6. **Verify the sm70 seams survived** -- the capability guard in
   `load_model_utils.py`, `SGL_KERNEL_V100_ONLY` in the AOT `CMakeLists.txt`,
   the `tilelang_fa_v100` selection in `platform_hook.py`.
7. **Re-run the runtime gates.** Static checks cannot see a wrong Triton merge.

### The trap that matters

Nothing in this fork fails loudly. The stock Marlin MoE kernel is an empty stub
below sm80 and writes zeros; JIT loaders report "unavailable" and fall back. A
merge that drops an sm70 branch produces a server that starts, answers, and is
wrong. Step 7 is not optional, and "it booted" is not step 7.

### What the static gates missed, both times

The 214-commit sync passed every static check — no conflict markers, no import
breakage, no deleted symbol referenced — and still failed to start twice. Both
failures share a cause worth internalising: **merge-ort does not touch
fork-only files.** When upstream renames an API and sweeps its own tree, our 67
fork-only files keep the old name, and nothing in a three-way merge notices.

1. `ForwardBatch.num_token_non_padded_cpu` -> `global_num_token_non_padded_cpu`
   (#37546). Upstream renamed every in-tree caller. `qwen4_exp.py` is fork-only,
   so it kept the old name and died with `AttributeError` on the first forward.
   An *attribute* access — invisible to an import sweep.

2. `resolve_spec_hidden_size` narrowed to DeepSeek-V4 only (#36805), because
   hy_v4 collapses its hc streams before the draft boundary. Qwen4-Exp does not,
   so its draft buffer silently came out 2560 wide instead of 10240 and draft
   graph capture failed on a `CHECK_EQ`. Nothing was renamed or deleted here —
   a *predicate got narrower*, and our model fell out of it.

The generalisable check, which does catch the first class:

```bash
# identifiers upstream removed, still referenced by fork-only files
comm -23 <(git ls-tree -r --name-only HEAD -- python/ | grep '\.py$' | sort) \
         <(git ls-tree -r --name-only <upstream> -- python/ | grep '\.py$' | sort)
```

The second class has no static check. It is why step 7 exists.

## What was worth having from these 214

Most are irrelevant to V100 (AMD/ROCm, NPU, XPU, diffusion, router, CI). The
ones that touch our paths:

- `a74470e90 fix(mamba): unify causal_conv1d col* dtype to x` — upstream landed
  the *same* fp16/bf16 fix the re-land had made independently. Convergent, and a
  useful signal that the fork's reading of that kernel was right.
- `07199fa22 [Performance] Optimize Qwen3.5 GDN prefill projection layouts` and
  `db89f639e [GDN] Amortize ReplaySSM checkpoint materialization` — the GDN
  prefill path, which is 36 of this model's 48 layers.
- `fe45af1e6 perf(gdn): select ReplaySSM verify loop unrolling by shape` — the
  MTP verify ring.
- `3c9cea8f1 [EAGLE] Prune draft-extend logits to selected rows` — the MTP path.
- `cf3173aeb [Perf] Walk the radix tree by offset` — radix-cache hot path.
- `4229088a4 feat(kernels): generalize persistent CuTe JIT cache` — could cut
  the sm70 JIT warm-up.
- `4372b8efa [1/N] Quantization Refactor: dedup the FP4 marlin helpers` — the
  one to watch. It deletes `marlin_make_empty_zp`,
  `prepare_moe_fp8_layer_for_marlin` and `MarlinConfig`. Nothing on the sm70
  path referenced them, but this is the class of change that removes a helper
  the fork depends on.

## Cost

Upstream moves ~70 commits/day. This sync — 214 commits, three days of drift —
was hours, and the bounded part (resolve + static gates) was well under one.
The original 4,250-commit re-land was days. Sync early; the cost is roughly
linear in commits that land on the sm70 surface, and the surface is small.
