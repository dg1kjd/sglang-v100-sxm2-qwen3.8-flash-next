# Known issues and open work

Open items, stated plainly. Ordered by what matters most to someone running
this. Nothing here blocks serving Qwen3.8-Flash-Next; each is a rough edge with
a known cause.

Status: `[ ]` open, `[x]` done, `[~]` done in source but unverified at runtime.

## Build / packaging — matters most if this is published

- [x] **FA3 target ignored `SGL_KERNEL_V100_ONLY`.** *(fixed and verified)* `DEFAULT_SGL_KERNEL_ENABLE_FA3`
  turns ON for any CUDA >= 12.4 and the `flash_ops` target was gated by
  `if (SGL_KERNEL_ENABLE_FA3)` alone, while `SGL_KERNEL_V100_ONLY` only stripped
  `compute_90a` from the *other* targets. A "V100-only" build therefore compiled
  the full Hopper FlashAttention-3 kernel set -- the single largest chunk of the
  build -- which cannot run on sm70.
  Fixed in `python/sglang/kernels/aot/CMakeLists.txt` (added
  `AND NOT SGL_KERNEL_V100_ONLY`). Safe because `sgl_kernel.flash_attn` is
  imported only from `_load_fa3_kernel_from_sgl()`, behind the `fa3` attention
  backend, and nothing under `srt/` references `flash_attention_v3` at all.
  **Measured:** the build plan dropped from 414 objects to 38 (-90%), and a
  full clean build went from 60+ min to **10m56s**. Wheel is 4 MB.
  Not a merge regression -- the fork's own CMakeLists had the same gap.

- [ ] **`scripts/install_v100.sh` provisions the wrong toolchain.** It installs
  CUDA 12.8 and `g++-12`; this host has CUDA 12.9, and its default `gcc` is 15
  with no `cc1plus` while only `g++-14` is installed. CUDA 12.9's
  `host_config.h` caps support at GCC 14, so 14 is both the working and the only
  supported choice. The correct recipe is already encoded in
  `docs/v100/INSTALL.md` §5 (pins `CC`/`CXX`/`CUDAHOSTCXX`);
  fold it back, and prefer auto-detecting the newest GCC <= 14 that actually has
  a `cc1plus` over hardcoding a version.

- [ ] **`scripts/ci/utils/compute_partitions.py`** still globs
  `python/sglang/jit_kernel/{tests,benchmark}`, which RFC #29630 retired. The
  fork never touched this file -- it is upstream's own stale reference, so fix
  it upstream rather than carrying a fork delta.

## Correctness / completeness

- [ ] **Phase 2: re-land `multimodal_gen`.** Phase 1 resolved all 92 of its
  conflicts to upstream, so the fork's V100 vision-port delta is currently *not*
  applied, and its fork-only files sit beside upstream's versions. `srt/` never
  imports `multimodal_gen`, so this cannot affect LLM serving -- but the import
  sweep's remaining 30 missing symbols are all here, and §9.5's vision smoke
  depends on it.

- [ ] **The two deferred sm70 startup warmups.** `_warmup_sm70_flashinfer_sampling`
  (~55 L) serialises the FlashInfer sampling-module build so four TP ranks do not
  race four Ninja builds on a cold cache; `_warmup_prefill_kernels_extends`
  (~130 L) compiles the prefill kernels at startup instead of on the first
  request. Both are startup-latency, not correctness. Deferred because
  `DecodeInputBuffers.create` no longer exists upstream and `ForwardBatch`'s
  constructor drifted, making a blind port unverifiable. Re-entry point:
  upstream's `ModelRunner.prewarm_sampling()` override hook, with the bodies in a
  new `model_runner_components/sm70_warmup.py` taking narrow keyword args.
  **Trigger:** a Ninja race across TP ranks at startup, or first-request latency
  materially worse than the §4 baseline.

- [ ] **`SGLANG_V100_GREEDY_TP_TOP1` is half-wired.** `logits_processor.py` still
  produces `logits_output.greedy_token_ids`, but the consumer fast path in
  `ModelRunner.sample()` was dropped (domain logic in a frozen file, and the env
  var defaults off). Inert today; either restore the consumer as a
  guard+delegate+return, or drop the producer. Do not leave it half-wired --
  enabling the env var would then silently pay for an all-gather nobody reads.

- [ ] **Register the remaining ~26 `SGLANG_SM70_*` / `SGLANG_V100_*` env vars.**
  Only `SGLANG_SM70_FORCE_FP16` and `SGLANG_CUSTOM_ALLREDUCE_ALGO` were moved
  into `Envs`; the rest are still raw `os.environ.get`, a Rule 1 violation under
  `.claude/skills/env-var-conventions`. Full inventory with read sites and
  defaults: the SM70/V100 section of `python/sglang/srt/environ.py`.

## Verification debt (from the resolution agents)

- [ ] **Dense NVFP4 linear layers may have no sm70 GEMM** after upstream's
  `fp4_gemm` rewrite. Only matters if the checkpoint NVFP4-quantises anything
  outside the experts. (quant/moe agent, UNVERIFIED-1.)
- [ ] **Draft-extend CUDA-graph capture for compressed QSA** -- unverified
  interaction. (speculative agent, §5.)
- [ ] **sm70 import-safety of the `dsa` / `deepseek_v4` / `flashmla` backends**,
  which upstream lazy-imports behind a bare `_is_cuda` guard -- and V100 *is*
  CUDA. (speculative agent, §6.)
- [ ] **`get_model_structural_tag(model="qwen_3_coder", ...)`** must still
  resolve for `scripts/smoke_v100.sh` after the xgrammar 0.1.32 -> 0.2.1 upgrade;
  that assert was written against the 0.1.32 key.

## Upstream sync

- [ ] **Next sync must handle upstream's tree-wide ruff-format first.**
  `28262c20d` (#37210) reformats 1,411 files; merging it head-on re-conflicts
  83 of the 167 files this re-land hand-resolved, for zero semantic gain. Run
  the formatter on our side as its own commit, then merge. Full recommendation
  and the list of substantive commits worth having: `docs/v100/UPSTREAM-SYNC.md`.

## Documentation

- [ ] The launcher's `--hicache-write-policy write_back` rationale cites
  `mem_cache/hi_mamba_radix_cache.py`, which upstream deleted (`12de7fb1f`,
  #33468). The behaviour still exists in `hiradix_cache.py` + `pool_host/mamba.py`;
  only the file reference is stale.
- [ ] `RELAND-PLAN.md` §2.5 lists `model_runner_kv_cache_mixin.py` as a hard
  engine file; upstream deleted it (`cfd17301a`, #31163). §5.0 records the
  correction, but the §2.5 list itself was left as the operator's measured record.

## Reporting

Issues with the sm70 path belong in this repository. Issues with SGLang itself
belong upstream at https://github.com/sgl-project/sglang.

Two bugs found during this port are upstream's, not this fork's, and are worth
reporting there:

- The Anthropic adapter collapses `output_config.effort="xhigh"` to `"max"` on
  the grounds that the OpenAI tier Literal lacks `xhigh`. It has included
  `xhigh` since `02236fa38`, so the conversion now only breaks templates that
  accept `xhigh` and not `max`.
- `SGL_KERNEL_V100_ONLY` does not gate the FlashAttention-3 target, so a
  "V100-only" build still compiles the full Hopper kernel set.
