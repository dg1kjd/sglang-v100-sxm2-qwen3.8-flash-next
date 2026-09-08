# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the per-platform backend defaults."""

from __future__ import annotations

import logging
import os
from typing import Any

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolving_view,
)
from sglang.srt.hardware_backend.mlx.runtime import use_mlx
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import get_device_sm, is_host_cpu_arm64

logger = logging.getLogger(__name__)


def handle_hardware_runtime_validation():
    # This is intentionally independent of `server_args.device`: setting
    # SGLANG_USE_MLX opts into the MLX backend and must fail immediately if
    # the environment cannot honor that request. With the flag unset,
    # use_mlx() remains lazy and does not import MLX.
    use_mlx()


def handle_npu_backends(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.device == "npu":
        from sglang.srt.hardware_backend.npu.utils import set_default_server_args

        set_default_server_args(server_args)

        current = cfg.cuda_graph_config.prefill.tc_compiler
        if current is not None and current != "eager":
            logger.warning(
                "At this moment Ascend platform only support prefill graph compilation with "
                "cuda_graph_config[prefill].tc_compiler='eager'."
            )
            declare_resolution(
                server_args,
                "_handle_npu_backends",
                cuda_graph_config=with_phase(
                    cfg.cuda_graph_config, Phase.PREFILL, tc_compiler="eager"
                ),
            )


def handle_mps_backends(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.device == "mps":
        if not use_mlx():
            declare_resolution(
                server_args,
                "_handle_mps_backends",
                disable_overlap_schedule=True,
            )


def handle_amd_specifics(server_args: Any):
    if get_platform().is_hip:
        declare_resolution(
            server_args, "_handle_amd_specifics", triton_attention_num_kv_splits=16
        )
        # Above this the HIP runtime registers a pageable H2D source with the
        # GPU rather than staging it, and the MMU notifier on that registration
        # evicts our KFD queues once per tensor while weights load. In KB.
        os.environ.setdefault("GPU_PINNED_MIN_XFER_SIZE", str(4 * 1024 * 1024))


def handle_nccl_pre_warm(server_args: Any):
    # pre_warm_nccl is only used with CUDA or HIP hardware or NPU hardware
    cfg = resolving_view(server_args)
    if cfg.pre_warm_nccl and not (
        get_platform().is_cuda or get_platform().is_hip or get_platform().is_npu
    ):
        logger.warning(
            "pre_warm_nccl is only applicable for CUDA or HIP hardware or NPU hardware. "
            "Ignoring pre_warm_nccl setting on current hardware."
        )
        declare_resolution(server_args, "_handle_nccl_pre_warm", pre_warm_nccl=False)


def handle_symm_mem_device_support(server_args: Any):
    cfg = resolving_view(server_args)
    # The symm-mem allocator compiles a CUDA plugin and links -lnccl, so off
    # CUDA/HIP (e.g. Ascend NPU) it fails deep in a build step rather than here.
    if cfg.enable_symm_mem and not (get_platform().is_cuda or get_platform().is_hip):
        logger.warning(
            "--enable-symm-mem is not supported on non CUDA/HIP devices "
            "(NCCL symmetric memory is unavailable). Disabling symmetric memory."
        )
        declare_resolution(
            server_args, "_handle_symm_mem_device_support", enable_symm_mem=False
        )


def handle_xpu_backends(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.device == "xpu":
        # Decode graph is opt-in on XPU: unless the user explicitly set
        # --cuda-graph-backend-decode (or --cuda-graph-config), keep it
        # disabled so the default startup doesn't require graph capture.
        if (Phase.DECODE, "backend") not in server_args._cuda_graph_config_locked:
            declare_resolution(
                server_args,
                "_handle_xpu_backends",
                cuda_graph_config=with_phase(
                    cfg.cuda_graph_config, Phase.DECODE, backend=Backend.DISABLED
                ),
            )
        elif cfg.cuda_graph_config.decode.backend not in (
            Backend.DISABLED,
            Backend.FULL,
        ):
            logger.warning(
                "XPU platform only supports decode backend 'full'; "
                "disabling unsupported decode backend '%s'.",
                cfg.cuda_graph_config.decode.backend,
            )
            declare_resolution(
                server_args,
                "_handle_xpu_backends",
                cuda_graph_config=with_phase(
                    cfg.cuda_graph_config, Phase.DECODE, backend=Backend.DISABLED
                ),
            )


def handle_sm70_backends(server_args: Any):
    # V100 (sm70) serves on SGLang's packaged TileLang attention kernels; no
    # separately installed FlashAttention-V100 extension is required.
    if not get_platform().is_cuda or get_device_sm() != 70:
        return
    cfg = resolving_view(server_args)
    if cfg.attention_backend is None:
        try:
            import tilelang  # noqa: F401

            from sglang.srt.layers.attention.tilelang_fa_v100 import (  # noqa: F401
                paged_forward,
            )
        except Exception:
            return
        declare_resolution(
            server_args,
            "_handle_sm70_backends",
            attention_backend="tilelang_fa_v100",
        )
        logger.info("SM70 (V100): auto-selecting 'tilelang_fa_v100'.")
    # The TileLang paged kernels index KV in 16-token pages; the launcher passes
    # the backend explicitly and no --page-size, so this is the live path.
    if (
        cfg.attention_backend in ("tilelang_fa_v100", "flash_attn_v100")
        and cfg.page_size is None
    ):
        declare_resolution(server_args, "_handle_sm70_backends", page_size=16)
        logger.info("SM70 (V100): auto-setting page_size=16.")


def handle_cpu_backends(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.device == "cpu":
        if cfg.attention_backend is None:
            declare_resolution(
                server_args,
                "_handle_cpu_backends",
                attention_backend=(
                    "torch_native" if is_host_cpu_arm64() else "intel_amx"
                ),
            )
        declare_resolution(
            server_args,
            "_handle_cpu_backends",
            sampling_backend="pytorch",
        )


def handle_hpu_backends(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.device == "hpu":
        declare_resolution(
            server_args,
            "_handle_hpu_backends",
            attention_backend="torch_native",
        )
        declare_resolution(
            server_args,
            "_handle_hpu_backends",
            sampling_backend="pytorch",
        )
