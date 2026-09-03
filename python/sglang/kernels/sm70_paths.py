"""Where the SM70 (V100) kernel artifacts live.

Two kinds, both of which sat under ``sglang/jit_kernel/`` before upstream retired
that package in RFC #29630:

- prebuilt extension modules (``_sm70_turbomind_v100.so`` and the two Marlin
  ``.so``), which are large out-of-band build outputs and are untracked --
  ``.gitignore`` excludes ``*.so``, so git will not move them for you;
- CUDA sources compiled on demand, which moved with the rest of the JIT tree.

This module is the single place that knows either location. It exists because
six call sites independently computed ``<...>/jit_kernel``, and every one of
them fails *silently*: the loaders report the kernel as unavailable rather than
raising, and the stock Marlin MoE path is an empty stub on SM70 that writes
nothing -- so a missed path produces wrong output, not a crash.
"""

from __future__ import annotations

import os
from pathlib import Path

_KERNELS_ROOT = Path(__file__).resolve().parent

# Overridable because container images install the prebuilt artifacts outside
# the source checkout (see docker/v100.Dockerfile).
_PREBUILT_DIR_ENV = "SGLANG_SM70_PREBUILT_DIR"


def sm70_prebuilt_dir() -> Path:
    """Directory holding the prebuilt SM70 ``.so`` artifacts."""
    override = os.environ.get(_PREBUILT_DIR_ENV)
    return Path(override) if override else _KERNELS_ROOT / "prebuilt"


def sm70_prebuilt(name: str) -> Path:
    return sm70_prebuilt_dir() / name


def sm70_csrc(*parts: str) -> Path:
    """A CUDA source under the JIT tree, e.g. ``sm70_csrc("sm70_longctx_decode.cu")``."""
    return _KERNELS_ROOT.joinpath("jit", "csrc", *parts)
