"""CPU tests for official DeepSeek-V4.1-Flash checkpoint names and v1 config.

No 476 GiB load. Index scan runs only when model.safetensors.index.json is
already on disk (download in flight is fine).
"""

from __future__ import annotations

import json
import os

from sglang.srt.configs.deepseek_v41 import (
    DeepseekV41Config,
    normalize_deepseek_v41_config,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.mem_cache.dsv41_v100_cpu_mock import (
    OFFICIAL_CONFIG,
    _REMAP_CASES,
    check_index_if_present,
    check_weight_remap,
    run_cpu_mock,
)
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    _skip_dsv41_language_only_weight,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class TestDsv41OfficialCheckpointNames(CustomTestCase):
    def test_inference_format_remap_matches_sglang_params(self):
        for src, want in _REMAP_CASES:
            self.assertEqual(
                DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(src),
                want,
                src,
            )

    def test_gate_bias_vl_is_not_mangled(self):
        name = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(
            "layers.6.ffn.gate.bias_vl"
        )
        self.assertEqual(name, "model.layers.6.mlp.gate.bias_vl")
        self.assertTrue(_skip_dsv41_language_only_weight(name))

    def test_engram_embed_keeps_scale_wkv_becomes_weight_scale_inv(self):
        embed = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(
            "layers.14.engram.embed.scale"
        )
        wkv = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(
            "layers.14.engram.wkv.scale"
        )
        self.assertEqual(embed, "model.layers.14.engram.embed.scale")
        self.assertEqual(wkv, "model.layers.14.engram.wkv.weight_scale_inv")

    def test_language_only_skips_vision(self):
        self.assertTrue(
            _skip_dsv41_language_only_weight("vision.patch_embed.proj.weight")
        )
        self.assertFalse(
            _skip_dsv41_language_only_weight("model.layers.6.mlp.gate.weight")
        )

    def test_index_scan_when_downloaded(self):
        lines = []
        code = check_index_if_present(lines)
        self.assertIsNone(code, msg="\n".join(lines))


class TestDsv41OfficialConfig(CustomTestCase):
    def test_normalize_remaps_architecture(self):
        values = normalize_deepseek_v41_config(
            {
                "architectures": ["DeepseekV41ForCausalLM"],
                "text_config": {
                    "hidden_size": 5120,
                    "n_routed_experts": 384,
                    "num_hidden_layers": 40,
                },
            }
        )
        self.assertEqual(values["architectures"], ["DeepseekV4ForCausalLM"])
        self.assertEqual(values["hidden_size"], 5120)
        self.assertEqual(values["n_routed_experts"], 384)

    def test_fp8_config_official_quant_block(self):
        cfg = Fp8Config.from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [32, 32],
                "scale_fmt": "ue8m0",
                "expert_dtype": "fp4",
            }
        )
        self.assertTrue(cfg.use_mxfp8)
        self.assertTrue(cfg.is_fp4_experts)
        self.assertEqual(list(cfg.weight_block_size), [32, 32])

    def test_on_disk_config_if_present(self):
        if not os.path.isfile(OFFICIAL_CONFIG):
            self.skipTest(f"official config not at {OFFICIAL_CONFIG}")
        raw = json.loads(open(OFFICIAL_CONFIG).read())
        hf = DeepseekV41Config(**raw)
        self.assertEqual(hf.hidden_size, 5120)
        self.assertEqual(hf.n_routed_experts, 384)
        self.assertEqual(hf.moe_intermediate_size, 2304)
        self.assertEqual(tuple(hf.engram_layer_ids), (1, 14))
        self.assertEqual(hf.num_experts_per_tok, 6)


class TestDsv41CpuMock(CustomTestCase):
    def test_cpu_mock_v1_shape(self):
        report = run_cpu_mock()
        self.assertTrue(report.ok, msg="\n".join(report.lines))
        self.assertEqual(report.exit_code, 0)

    def test_weight_remap_helper(self):
        lines = []
        self.assertIsNone(check_weight_remap(lines), msg="\n".join(lines))
