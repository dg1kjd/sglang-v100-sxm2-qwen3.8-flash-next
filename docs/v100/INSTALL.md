# Building SGLang V100 from source

Everything here has been executed end-to-end on the reference host. Where a step
has a common failure mode, it is written down with the exact error text, because
most of these fail *quietly* — a missing sm70 kernel does not raise, it returns
"unavailable" and silently falls back to a path that produces wrong output.

`scripts/install_v100.sh` automates the whole sequence. This document explains
what it does, so you can run the steps individually when one fails.

---

## 0. Prerequisites

### Hardware

- 4× V100 32 GB. SXM2 preferred; NVLink helps but a partial mesh is fine (the
  custom all-reduce is pinned to single-stage for exactly that case).
- 340 GB+ host RAM. The PLE n-gram table is host-resident and the hierarchical
  KV cache wants the rest.
- ~250 GB disk for the model, plus room for the disk cache tier.

### CUDA

**CUDA 12.8 or 12.9. Not 13.x** — CUDA 13 removed support for compute
capabilities below 7.5, which includes Volta. There is no workaround.

```bash
/usr/local/cuda/bin/nvcc --version   # expect 12.8 or 12.9
```

### Host compiler — the one that catches everyone

CUDA 12.9 caps host-compiler support at **GCC 14**, and its own header says so:

```c
#if __GNUC__ > 14
#error -- unsupported GNU version! gcc versions later than 14 are not supported!
```

Many current distributions default `gcc` to 15. Worse, some install `gcc-15`
without `g++-15`, so the default driver has no `cc1plus` at all and every CUDA
compile dies with:

```
gcc: fatal error: cannot execute 'cc1plus': posix_spawnp: No such file or directory
nvcc fatal   : Failed to preprocess host compiler properties.
```

Check and pin:

```bash
ls /usr/libexec/gcc/x86_64-linux-gnu/*/cc1plus     # which GCCs are complete
g++-14 --version                                   # install if absent

export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14
export CUDAHOSTCXX=/usr/bin/g++-14
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-14"
```

`NVCC_PREPEND_FLAGS` matters separately from `CUDAHOSTCXX`: the runtime JIT
kernels invoke a bare `nvcc`, which reads the former but not the latter. The
serve script exports both.

---

## 1. Clone

```bash
git clone https://github.com/haohervchb/sglang-V100.git
cd sglang-V100
```

## 2. Python environment

Python 3.12. Any venv tool works; the reference host uses `uv`.

```bash
uv venv --python 3.12 ~/sglang-v100-venv
source ~/sglang-v100-venv/bin/activate
```

Keep the venv's `bin` on `PATH` for everything below — the JIT kernels shell out
to `ninja`, which is a venv-local binary with no system copy.

## 3. Python dependencies

```bash
pip install --no-deps -r requirements.txt
```

`--no-deps` is mandatory. `python/pyproject.toml` carries upstream's pins, which
target the CUDA 13 stack; letting pip resolve freely pulls cu13 wheels and
breaks Volta support. `requirements.txt` is the validated cu12 floor.

Then install SGLang itself, still without dependency resolution:

```bash
pip install --no-deps -e python/
```

## 4. FlashInfer for sm70

Upstream FlashInfer does not build for Volta. The port carries a patch:

```bash
export SGLANG_V100_DEPS_DIR=~/.cache/sglang-v100-sources
# clone at the pinned revision, apply patches/flashinfer-sm70.patch, then:
pip install --no-deps --no-build-isolation -e "$SGLANG_V100_DEPS_DIR/flashinfer-sm70"
```

Pinned revision: `c3c40a7b90b792fc59f90f8f55c9e2de9c1b6833`.
`scripts/install_v100.sh` does the clone-and-patch for you.

## 5. sglang-kernel, sm70 only

```bash
export TORCH_CUDA_ARCH_LIST=7.0
export CMAKE_ARGS="-DSGL_KERNEL_V100_ONLY=ON -DSGL_KERNEL_COMPILE_THREADS=2"
pip install --no-deps --no-build-isolation python/sglang/kernels/aot
```

Two things to know:

- **`--no-build-isolation` is required, not an optimisation.** The AOT
  `pyproject.toml` declares `torch==2.13.0` as a *build* requirement. An
  isolated build would fetch that and produce an extension whose ABI does not
  match the runtime's torch 2.9.1.
- **`SGL_KERNEL_V100_ONLY=ON` matters a lot.** With it, the build is 38 objects
  and about 11 minutes. Without it, the Hopper FlashAttention-3 target is also
  compiled — 374 more objects of sm90a kernels that cannot run on Volta,
  turning an 11-minute build into an hour.

Confirm the arch in the configure output:

```
-- Added CUDA NVCC flags for: -gencode;arch=compute_70,code=sm_70
```

(The CMake target is named `common_ops_sm100_build` regardless — that is a
label, not the architecture.)

## 6. TurboMind sm70 and Marlin V100

```bash
python scripts/build_sm70_turbomind.py     # block-FP8 + FP16 MoE backend
bash   scripts/setup_v100_marlin.sh        # GPTQ/AWQ repack + NVFP4 MoE
```

Both install their `.so` into `python/sglang/kernels/prebuilt/`, which is where
`sglang.kernels.sm70_paths` looks. That module is the single place that knows
the location; if you relocate the artifacts, change it there and nowhere else.

**This step is not optional.** The stock upstream Marlin MoE kernel is an empty
stub below sm80: it writes nothing. If the V100 kernels are missing, the server
starts, answers, and returns **zero-valued expert output** — plausible-looking
garbage rather than an error. You will see this warning at startup:

```
SM70 (V100) detected but the marlin_v100 MoE kernel was not found.
... routed-expert output will be ZERO (incorrect).
```

Treat it as fatal.

## 7. Restore the CUDA 12 NCCL

Some wheels pull the cu13 NCCL, which torch 2.9.1 cannot use:

```bash
pip uninstall -y nvidia-nccl-cu13
pip install --force-reinstall --no-deps nvidia-nccl-cu12==2.27.5
```

## 8. Model weights

Fetch the NVFP4 checkpoint of Qwen3.8-Flash-Next and point the launcher at it:

```bash
export FLASH_NEXT_MODEL=/path/to/Qwen3.8-Flash-Next-NVFP4
```

Sanity-check that it is the multimodal export if you want vision:

```bash
python -c "
import json; d=json.load(open('$FLASH_NEXT_MODEL/config.json'))
print('language_model_only:', d.get('language_model_only'))  # expect False
print('vision_config      :', bool(d.get('vision_config')))  # expect True
"
```

## 9. Verify

```bash
bash scripts/smoke_v100.sh
```

Expect every line to report a registration:

```
SGLang V100 environment is ready: 2.9.1+cu128
FlashInfer SM70 sampling: ...
Attention: SGLang TileLang SM70 package
SM70 kernel: .../sgl_kernel/sm70/common_ops.abi3.so
SM70 Marlin repack: registered
SM70 TurboMind FP8: registered
SM70 TurboMind FP16 MoE: registered
SM70 TurboMind exact AWQ dequantizer: registered
NCCL: 2.27.5
```

If `SGLANG_V100_PYTHON` is not on `PATH`, set it explicitly:

```bash
SGLANG_V100_PYTHON=$(which python) bash scripts/smoke_v100.sh
```

## 10. Serve

```bash
bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp
```

First launch compiles the sm70 JIT kernels; expect several minutes, and expect
all four ranks to compile at once. Later launches reuse `~/.cache/sglang/jit/sm70`.

Ready when the log says:

```
The server is fired up and ready to roll!
```

---

## Troubleshooting

Failures observed during bring-up, with their causes.

| Symptom | Cause |
|---|---|
| `cannot execute 'cc1plus'` | Default `gcc` is 15 (or has no C++ backend). Set `CC`/`CXX`/`CUDAHOSTCXX`/`NVCC_PREPEND_FLAGS` to GCC 14 — see §0. |
| `No such file or directory: 'ninja'` | The venv's `bin` is not on `PATH`. The JIT shells out to `ninja`, and there is usually no system copy. |
| `sglang-kernel is installed with version 0.4.3, which is less than 0.4.6.post1` | Rebuild step 5 from `python/sglang/kernels/aot` (the package moved out of the repo root). |
| `cannot import name 'AnyTokensFormat' from 'xgrammar.structural_tag'` | xgrammar is older than 0.2.1. |
| `SM70 NVFP4 decode extension is unavailable` | A relocated `.cu` was not found. All sm70 sources live under `python/sglang/kernels/jit/csrc/`; consumers must resolve them via `sglang.kernels.sm70_paths.sm70_csrc()`. |
| `ModuleNotFoundError: No module named 'deep_gemm'` during graph capture | An sm90+ backend was imported behind a bare `_is_cuda` gate — and V100 *is* CUDA. Optional backends must load through `_optional_graph_backend_types()`. |
| `AssertionError: Unsupported layout: layer_first` | `--hicache-mem-layout` must be `page_first`. `MambaPoolHost` accepts only `page_first`/`page_first_direct`. |
| `slice [0, N) escapes the M-byte pull workspace` | An all-reduce larger than the workspace was forced into the custom path. Oversized reduces must fall back to NCCL. |
| Correct-looking but nonsensical answers | Almost certainly the Marlin MoE stub — see §6. Check the startup log for the ZERO-output warning. |
| ~100% CPU per rank while idle | `--sleep-on-idle` is not in effect. Healthy idle is ~4% per rank, with the scheduler blocked in `zmq.poll`. |

### Build takes an hour

You forgot `-DSGL_KERNEL_V100_ONLY=ON`, and are compiling the Hopper FA3 kernel
set. See §5.
