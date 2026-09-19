# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""BreakableCudaGraphBackend — segment-captured graphs with eager break
markers (eager_on_graph decorators on attention / mamba layers).
No torch.compile.
"""

from __future__ import annotations

import dataclasses
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin import (
    DedupedCudaGraphMixin,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
    enable_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_utils.pool import (
    GraphPoolPrecarve,
    get_or_create_global_graph_memory_pool,
    graph_pool_capture_scope,
    graph_pool_replay_scope,
)
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey

logger = logging.getLogger(__name__)


def _eager_clone_tensor(t: Any) -> Any:
    if not torch.is_tensor(t) or t.numel() == 0:
        return t
    out = torch.empty(t.shape, dtype=t.dtype, device=t.device)
    out.copy_(t)
    return out


def _eager_clone_lpo(output: LogitsProcessorOutput) -> LogitsProcessorOutput:
    kwargs = {}
    for field in dataclasses.fields(output):
        val = getattr(output, field.name)
        kwargs[field.name] = _eager_clone_tensor(val) if torch.is_tensor(val) else val
    return LogitsProcessorOutput(**kwargs)


def select_bcg_replay_output(live: Any, stored: Any) -> Any:
    """Pick the model return after a segmented replay.

    ``BreakableCUDAGraph.replay`` returns the last eager-break value. Target
    decode's last break is the logits processor (an LPO). DSpark draft's last
    break is CSA2/HC (a Tensor) while the captured return is an LPO of
    ``hidden_states``. Prefer a live LPO, else the stored LPO.
    """
    if isinstance(live, LogitsProcessorOutput):
        return live
    if isinstance(stored, LogitsProcessorOutput):
        return stored
    return live if live is not None else stored


def _dsv41_log_logits(tag: str, out: Any) -> None:
    lg = getattr(out, "next_token_logits", None)
    gt = getattr(out, "greedy_token_ids", None)
    hs = getattr(out, "hidden_states", None)
    lg_fin = None
    peek = None
    if torch.is_tensor(lg) and lg.numel():
        lg_fin = int(torch.isfinite(lg).sum().item())
        peek = float(lg.reshape(-1)[0].item())
    logger.info(
        "DSV41 BCG %s logits=%s greedy=%s hs=%s finite=%s/%s peek=%s",
        tag,
        None if lg is None else tuple(lg.shape),
        None if gt is None else tuple(gt.shape),
        None if not torch.is_tensor(hs) else tuple(hs.shape),
        lg_fin,
        None if not torch.is_tensor(lg) else lg.numel(),
        peek,
    )


class BreakableCudaGraphBackend(DedupedCudaGraphMixin, BaseCudaGraphBackend):
    """Segmented capture: graphs break at attention / mamba boundaries;
    attention metadata is recomputed at replay outside captured segments.
    """

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
        debug_eager: bool = False,
    ) -> None:
        self._model_runner = cuda_graph_runner.model_runner
        self._graphs: Dict[Any, BreakableCUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._capture_inputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._debug_eager = debug_eager
        self._shared_output_buffer: Optional[Any] = None
        self._precarve = GraphPoolPrecarve()
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            raise NotImplementedError(
                "Breakable CUDA graph is not compatible with memory saver mode"
            )

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        if self._pool is None:
            self._pool = get_or_create_global_graph_memory_pool(self._device_module)
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        self._shared_output_buffer = None
        self.begin_cuda_graph_capture()
        try:
            with self.replay_session():
                yield
        finally:
            try:
                self.end_cuda_graph_capture()
            finally:
                self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        warmup_out = None
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            with self._precarve.measure():
                warmup_out = forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        graph = BreakableCUDAGraph(self.deduped_cuda_graph)
        captured_fn = (
            eager_on_graph(True)(forward_fn) if self._debug_eager else forward_fn
        )
        size = shape_key.size
        if self._shared_output_buffer is None:
            self._shared_output_buffer = self._alloc_full_buffer(warmup_out, size)
        with (
            graph_pool_capture_scope(),
            BreakableCUDAGraphCapture(
                cuda_graph=graph,
                pool=self._pool,
                stream=self._capture_stream,
                barrier_fn=self._tp_group.barrier,
            ),
        ):
            self._precarve.mint()
            out = captured_fn()

        # Do not capture the output-buffer copy. Eager breaks already write
        # into ``out`` (logits LPO); a captured copy was the last graph
        # segment and hid IndexKernel OOBs until after replay() returned.
        if post_warmup_hook is not None:
            post_warmup_hook()

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out
        # CUDA graphs retain tensor addresses, not Python tensor lifetimes.
        self._capture_inputs[shape_key] = capture_inputs

    @staticmethod
    def _map_logits_tensors(
        output: LogitsProcessorOutput, fn: Callable[[torch.Tensor], Any]
    ) -> LogitsProcessorOutput:
        """Resize or slice tensor fields; leave Python-side logprob lists as-is."""
        kwargs = {}
        for field in dataclasses.fields(output):
            val = getattr(output, field.name)
            kwargs[field.name] = fn(val) if torch.is_tensor(val) else val
        return LogitsProcessorOutput(**kwargs)

    def _output_rows(self, output: Any, cap: int) -> int:
        """Leading-dim row count actually produced by the body, clamped to ``cap``.

        A body that shards or prunes its output along dim 0 returns fewer than
        ``cap`` rows; everything else returns exactly ``cap``.
        """
        if torch.is_tensor(output):
            return min(cap, output.shape[0])
        if isinstance(output, LogitsProcessorOutput):
            logits = output.next_token_logits
            if logits is None:
                return cap
            return min(cap, logits.shape[0])
        if isinstance(output, PPProxyTensors):
            rows = [t.shape[0] for t in output.tensors.values()]
            return min([cap, *rows])
        if isinstance(output, (list, tuple)) and output:
            return min(self._output_rows(o, cap) for o in output if o is not None)
        return cap

    def _alloc_full_buffer(self, output: Any, size: int) -> Any:
        """A same-structure buffer as ``output`` but with ``size`` leading rows."""
        if output is None:
            return None
        if torch.is_tensor(output):
            return output.new_empty((size, *output.shape[1:]))
        if isinstance(output, LogitsProcessorOutput):
            return self._map_logits_tensors(
                output, lambda t: self._alloc_full_buffer(t, size)
            )
        if isinstance(output, PPProxyTensors):
            return PPProxyTensors(
                {
                    key: t.new_empty((size, *t.shape[1:]))
                    for key, t in output.tensors.items()
                }
            )
        if isinstance(output, tuple):
            return tuple(self._alloc_full_buffer(o, size) for o in output)
        if isinstance(output, list):
            return [self._alloc_full_buffer(o, size) for o in output]
        raise TypeError(f"Unsupported BCG output type: {type(output)}")

    def _slice_output(self, output: Any, num_tokens: int) -> Any:
        if output is None:
            return None
        if torch.is_tensor(output):
            return output[:num_tokens]
        if isinstance(output, LogitsProcessorOutput):
            return self._map_logits_tensors(
                output, lambda t: self._slice_output(t, num_tokens)
            )
        if isinstance(output, PPProxyTensors):
            return output[:num_tokens]
        if isinstance(output, tuple):
            return tuple(self._slice_output(item, num_tokens) for item in output)
        if isinstance(output, list):
            return [self._slice_output(item, num_tokens) for item in output]
        raise TypeError(f"Unsupported BCG output type: {type(output)}")

    def _copy_output_to_buffer(
        self, output: Any, output_buffer: Any, num_tokens: int
    ) -> None:
        if output is None or output_buffer is None:
            if output is None and output_buffer is None:
                return
            raise ValueError(
                "BCG output structure changed between capture sizes: "
                f"{type(output)} vs {type(output_buffer)}"
            )
        if torch.is_tensor(output) and torch.is_tensor(output_buffer):
            output_buffer[:num_tokens].copy_(output[:num_tokens])
            return
        if isinstance(output, LogitsProcessorOutput) and isinstance(
            output_buffer, LogitsProcessorOutput
        ):
            for field in dataclasses.fields(output):
                src = getattr(output, field.name)
                if not torch.is_tensor(src):
                    continue
                self._copy_output_to_buffer(
                    src, getattr(output_buffer, field.name), num_tokens
                )
            return
        if isinstance(output, PPProxyTensors) and isinstance(
            output_buffer, PPProxyTensors
        ):
            if output.tensors.keys() != output_buffer.tensors.keys():
                raise ValueError(
                    "BCG output proxy structure changed between capture sizes: "
                    f"{output.tensors.keys()} != {output_buffer.tensors.keys()}"
                )
            for key, tensor in output.tensors.items():
                self._copy_output_to_buffer(
                    tensor, output_buffer.tensors[key], num_tokens
                )
            return
        if isinstance(output, (list, tuple)) and isinstance(
            output_buffer, type(output)
        ):
            if len(output) != len(output_buffer):
                raise ValueError(
                    "BCG output sequence structure changed between capture sizes: "
                    f"{len(output)} != {len(output_buffer)}"
                )
            for item, buffer in zip(output, output_buffer):
                self._copy_output_to_buffer(item, buffer, num_tokens)
            return
        raise TypeError(
            "Unsupported BCG output buffer pair: "
            f"{type(output)} vs {type(output_buffer)}"
        )

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        with enable_breakable_cuda_graph():
            yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        from sglang.srt.environ import envs

        sync = envs.SGLANG_DSV41_PREFILL_SYNC.get()
        with graph_pool_replay_scope():
            live = self._graphs[shape_key].replay()
            stored = self._outputs[shape_key]
            src = select_bcg_replay_output(live, stored)
            if sync:
                _dsv41_log_logits("inside pool", src)
            if isinstance(src, LogitsProcessorOutput):
                out = _eager_clone_lpo(src)
            elif torch.is_tensor(src):
                out = _eager_clone_tensor(src)
            else:
                out = src
        if sync:
            logger.info("DSV41 BCG backend after pool scope")
            torch.cuda.synchronize()
            idx = torch.zeros(1, device="cuda", dtype=torch.int64)
            torch.zeros(1, device="cuda")[idx]
            torch.cuda.synchronize()
            logger.info("DSV41 BCG backend after pool scope ok")
            _dsv41_log_logits("cloned", out)
            logger.info("DSV41 BCG out inspect ok")
        return out

    def cleanup(self) -> None:
        self.close()
        self._graphs.clear()
        self._outputs.clear()
        self._capture_inputs.clear()
        self._pool = None
        self._shared_output_buffer = None
