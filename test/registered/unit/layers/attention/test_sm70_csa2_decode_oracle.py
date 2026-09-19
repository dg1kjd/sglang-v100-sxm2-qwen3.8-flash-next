"""GPU oracle: D3 CSA2 prefill vs decode, hierarchy, and CUDA-graph decode."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.dsv4 import sm70_csa2_reference as R
from sglang.srt.layers.attention.dsv4.sm70_csa2 import (
    get_state,
    sm70_forward_low_ratio_sources,
    sm70_forward_sparse,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=90,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)

FP16 = torch.float16
ATOL = 2e-2
RTOL = 2e-2
LAYER_ORDER = (2, 20, 24)


def _require_sm70():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA required")
    if torch.cuda.get_device_capability()[0] != 7:
        raise unittest.SkipTest("SM70 required")


def _lin(v, weight):
    return (v.float() @ weight.float().t()).to(v.dtype)


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    sa = set(a[a >= 0].tolist())
    sb = set(b[b >= 0].tolist())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


class TestSm70Csa2DecodeOracle(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _require_sm70()
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        torch.set_num_threads(4)
        cls.dev = torch.device("cuda")
        g = torch.Generator(device="cpu").manual_seed(7)
        hidden, heads, q_lora_r, n_ih = 64, 4, 32, 8
        cls.hidden = hidden
        cls.heads = heads
        cls.q_lora_r = q_lora_r
        n_pos = 4096
        cls.freqs = R.precompute_freqs_cis(64, n_pos + 8, 0, 10000.0, 1.0, 32.0, 1.0)[
            :n_pos
        ].to(cls.dev)
        cls.wkv = (torch.randn(512, hidden, generator=g).to(FP16) * 0.02).to(cls.dev)
        cls.kv_norm = torch.ones(512, dtype=FP16, device=cls.dev)
        cls.wq_b = (torch.randn(n_ih * 128, q_lora_r, generator=g).to(FP16) * 0.02).to(
            cls.dev
        )
        cls.w_proj = (torch.randn(n_ih, hidden, generator=g).to(FP16) * 0.02).to(cls.dev)
        cls.wk = (torch.randn(128, 512, generator=g).to(FP16) * 0.02).to(cls.dev)
        cls.k_norm = torch.ones(128, dtype=FP16, device=cls.dev)
        cls.wkv_c = (torch.randn(512, hidden, generator=g).to(FP16) * 0.02).to(cls.dev)
        cls.wgate = (torch.randn(512, hidden, generator=g).to(FP16) * 0.02).to(cls.dev)
        cls.norm_c = torch.ones(512, dtype=FP16, device=cls.dev)
        cls.sink = torch.zeros(heads, dtype=torch.float32, device=cls.dev)
        t = 712
        cls.x = (torch.randn(t, hidden, generator=g).to(FP16) * 0.05).to(cls.dev)
        cls.q_lora = (torch.randn(t, q_lora_r, generator=g).to(FP16) * 0.05).to(cls.dev)
        cls.q = (torch.randn(t, heads, 512, generator=g).to(FP16) * 0.1).to(cls.dev)
        cls.positions = torch.arange(t, dtype=torch.int64, device=cls.dev)

    def _indexer(self, owns_k, is_src, uses_cand):
        return SimpleNamespace(
            owns_k=owns_k,
            is_candidate_source=is_src,
            uses_candidates=uses_cand,
            candidate_topk_blocks=2048,
            candidate_block_size=8,
            index_topk=512,
            wk=SimpleNamespace(weight=self.wk),
            k_norm=SimpleNamespace(weight=self.k_norm),
            queries=lambda ql, fr: R.fake_quant_fp4_ue8m0(
                R.rope_tail(
                    _lin(ql, self.wq_b).view(ql.shape[0], 8, 128), fr, 64
                )
            ),
            head_weights=lambda xx: _lin(xx, self.w_proj) * (128**-0.5 * 8**-0.5),
        )

    def _layer(self, lid, ratio, compressor, indexer):
        return SimpleNamespace(
            layer_id=lid,
            compress_ratio=ratio,
            head_dim=512,
            qk_rope_head_dim=64,
            eps=1e-20,
            sliding_window=128,
            softmax_scale=512**-0.5,
            freqs_cis=self.freqs,
            wkv=lambda v: (_lin(v, self.wkv), None),
            kv_norm=SimpleNamespace(weight=self.kv_norm),
            compressor=compressor,
            indexer=indexer,
            attn_sink=self.sink,
        )

    def _layers(self):
        c1 = SimpleNamespace(
            wkv=SimpleNamespace(weight=self.wkv_c),
            norm=SimpleNamespace(weight=self.norm_c),
        )
        c2 = SimpleNamespace(
            wkv=SimpleNamespace(weight=self.wkv_c),
            wgate=SimpleNamespace(weight=self.wgate),
            norm=SimpleNamespace(weight=self.norm_c),
        )
        l2 = self._layer(2, 2, c2, self._indexer(True, False, False))
        l20 = self._layer(20, 1, c1, self._indexer(True, True, False))
        l24 = self._layer(24, 1, None, self._indexer(False, False, True))
        return {2: l2, 20: l20, 24: l24}

    def _backend(self):
        return SimpleNamespace(model_runner=None, max_context_len=4096)

    def _fb(self, decode: bool, target_verify: bool = False):
        return SimpleNamespace(
            batch_size=1,
            forward_mode=SimpleNamespace(
                is_decode=lambda: decode and not target_verify,
                is_extend=lambda: (not decode) and (not target_verify),
                is_target_verify=lambda: target_verify,
            ),
        )

    def _run_extend(self, backend, layers, sl: slice):
        fb = self._fb(False)
        outs = {}
        for lid in LAYER_ORDER:
            sm70_forward_low_ratio_sources(
                backend,
                layers[lid],
                self.x[sl],
                self.q_lora[sl],
                self.positions[sl],
                fb,
            )
            outs[lid] = sm70_forward_sparse(
                backend, self.q[sl], layers[lid], fb, layers[lid].compress_ratio, self.sink
            )
        return outs

    def _run_decode(self, backend, layers, i: int):
        fb = self._fb(True)
        sl = slice(i, i + 1)
        outs = {}
        for lid in LAYER_ORDER:
            sm70_forward_low_ratio_sources(
                backend,
                layers[lid],
                self.x[sl],
                self.q_lora[sl],
                self.positions[sl],
                fb,
            )
            outs[lid] = sm70_forward_sparse(
                backend, self.q[sl], layers[lid], fb, layers[lid].compress_ratio, self.sink
            )
        return outs

    def _maxdiff(self, a, b):
        return float((a.float() - b.float()).abs().max().item())

    def _check_last(self, prefill_outs, decode_outs, tag: str):
        for lid in LAYER_ORDER:
            a = prefill_outs[lid][-1:]
            b = decode_outs[lid]
            d = self._maxdiff(a, b)
            print(f"{tag} L{lid} maxabs={d:.5f}")
            self.assertTrue(
                torch.allclose(a.float(), b.float(), atol=ATOL, rtol=RTOL),
                f"{tag} L{lid} maxabs={d}",
            )
            self.assertFalse(torch.isnan(b).any().item())

    def _scenario(self, t: int):
        layers_a = self._layers()
        be_a = self._backend()
        outs_a = self._run_extend(be_a, layers_a, slice(0, t))
        st_a = get_state(be_a)

        layers_b = self._layers()
        be_b = self._backend()
        self._run_extend(be_b, layers_b, slice(0, t - 1))
        outs_b = self._run_decode(be_b, layers_b, t - 1)
        st_b = get_state(be_b)
        self._check_last(outs_a, outs_b, f"T={t} a-vs-b")

        layers_c = self._layers()
        be_c = self._backend()
        if t - 1 > 301:
            self._run_extend(be_c, layers_c, slice(0, 301))
            self._run_extend(be_c, layers_c, slice(301, t - 1))
        else:
            self._run_extend(be_c, layers_c, slice(0, t - 1))
        outs_c = self._run_decode(be_c, layers_c, t - 1)
        self._check_last(
            {lid: outs_b[lid] for lid in LAYER_ORDER},
            outs_c,
            f"T={t} b-vs-c",
        )
        # b-vs-c compares decode to decode; reuse _check_last by faking prefill[-1:]
        for lid in LAYER_ORDER:
            d = self._maxdiff(outs_b[lid], outs_c[lid])
            self.assertTrue(
                torch.allclose(
                    outs_b[lid].float(), outs_c[lid].float(), atol=ATOL, rtol=RTOL
                ),
                f"T={t} b-vs-c L{lid} maxabs={d}",
            )

        extra = 3
        print(
            f"T={t} L20 topk jaccard={_jaccard(st_b.topk[20][0], st_a.prefill_topk[20][-1]):.4f} "
            f"cand jaccard={_jaccard(st_b.cand_ids[0], st_a.prefill_cand_ids[-1]):.4f} "
            f"L24 topk jaccard={_jaccard(st_b.topk[24][0], st_a.prefill_topk[24][-1]):.4f}"
        )
        self.assertGreaterEqual(
            _jaccard(st_b.topk[20][0], st_a.prefill_topk[20][-1]), 0.95
        )
        self.assertGreaterEqual(
            _jaccard(st_b.cand_ids[0], st_a.prefill_cand_ids[-1]), 0.95
        )
        self.assertGreaterEqual(
            _jaccard(st_b.topk[24][0], st_a.prefill_topk[24][-1]), 0.95
        )
        for i in range(t, t + extra):
            more = self._run_decode(be_b, layers_b, i)
            for lid in LAYER_ORDER:
                self.assertTrue(torch.isfinite(more[lid].float()).all().item())
        layers_d = self._layers()
        be_d = self._backend()
        outs_d = self._run_extend(be_d, layers_d, slice(0, t + extra))
        layers_e = self._layers()
        be_e = self._backend()
        self._run_extend(be_e, layers_e, slice(0, t + extra - 1))
        outs_e = self._run_decode(be_e, layers_e, t + extra - 1)
        self._check_last(outs_d, outs_e, f"T={t}+3 a-vs-decode")

    def test_prefill_vs_decode_t700_and_t9(self):
        self._scenario(700)
        self._scenario(9)

    def test_decode_cuda_graph_capture(self):
        t = 64
        fb = self._fb(True)
        static_x = self.x[t : t + 1].clone()
        static_ql = self.q_lora[t : t + 1].clone()
        static_q = self.q[t : t + 1].clone()
        static_pos = self.positions[t : t + 1].clone()

        def warmup_to_tm1(backend, layers):
            self._run_extend(backend, layers, slice(0, t - 1))
            static_x.copy_(self.x[t - 1 : t])
            static_ql.copy_(self.q_lora[t - 1 : t])
            static_q.copy_(self.q[t - 1 : t])
            static_pos.copy_(self.positions[t - 1 : t])
            for lid in LAYER_ORDER:
                sm70_forward_low_ratio_sources(
                    backend, layers[lid], static_x, static_ql, static_pos, fb
                )
                sm70_forward_sparse(
                    backend, static_q, layers[lid], fb, layers[lid].compress_ratio, self.sink
                )

        layers_ref = self._layers()
        backend_ref = self._backend()
        warmup_to_tm1(backend_ref, layers_ref)
        ref = self._run_decode(backend_ref, layers_ref, t)

        layers = self._layers()
        backend = self._backend()
        warmup_to_tm1(backend, layers)
        static_x.copy_(self.x[t : t + 1])
        static_ql.copy_(self.q_lora[t : t + 1])
        static_q.copy_(self.q[t : t + 1])
        static_pos.copy_(self.positions[t : t + 1])
        st = get_state(backend)
        snap = {
            "swa": {k: v.clone() for k, v in st.swa_ring.items()},
            "kv": {k: v.clone() for k, v in st.kv_rows.items()},
            "ix": {k: v.clone() for k, v in st.index_rows.items()},
            "pkv": {k: v.clone() for k, v in st.pending_kv.items()},
            "psc": {k: v.clone() for k, v in st.pending_score.items()},
            "topk": {k: v.clone() for k, v in st.topk.items()},
            "cand": None if st.cand_ids is None else st.cand_ids.clone(),
            "ck": None if st.cand_key_ids is None else st.cand_key_ids.clone(),
        }

        def restore():
            for k, v in snap["swa"].items():
                st.swa_ring[k].copy_(v)
            for k, v in snap["kv"].items():
                st.kv_rows[k].copy_(v)
            for k, v in snap["ix"].items():
                st.index_rows[k].copy_(v)
            for k, v in snap["pkv"].items():
                st.pending_kv[k].copy_(v)
            for k, v in snap["psc"].items():
                st.pending_score[k].copy_(v)
            for k, v in snap["topk"].items():
                st.topk[k].copy_(v)
            if snap["cand"] is not None:
                st.cand_ids.copy_(snap["cand"])
            if snap["ck"] is not None:
                st.cand_key_ids.copy_(snap["ck"])

        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        captured = {}
        with torch.cuda.graph(g):
            for lid in LAYER_ORDER:
                sm70_forward_low_ratio_sources(
                    backend, layers[lid], static_x, static_ql, static_pos, fb
                )
                captured[lid] = sm70_forward_sparse(
                    backend,
                    static_q,
                    layers[lid],
                    fb,
                    layers[lid].compress_ratio,
                    self.sink,
                )
        restore()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        for lid in LAYER_ORDER:
            d = self._maxdiff(captured[lid], ref[lid])
            print(f"graph replay L{lid} maxabs={d:.5f}")
            self.assertTrue(
                torch.allclose(
                    captured[lid].float(), ref[lid].float(), atol=1e-3, rtol=1e-3
                ),
                f"graph L{lid} maxabs={d}",
            )

    def _run_verify(self, backend, layers, sl: slice):
        fb = self._fb(False, target_verify=True)
        outs = {}
        for lid in LAYER_ORDER:
            sm70_forward_low_ratio_sources(
                backend,
                layers[lid],
                self.x[sl],
                self.q_lora[sl],
                self.positions[sl],
                fb,
            )
            outs[lid] = sm70_forward_sparse(
                backend, self.q[sl], layers[lid], fb, layers[lid].compress_ratio, self.sink
            )
        return outs

    def test_target_verify_t6_matches_prefill_window(self):
        t0, t = 64, 6
        backend = self._backend()
        backend.model_runner = SimpleNamespace(decode_num_tokens_per_req=lambda: 6)
        layers = self._layers()
        pref = self._run_extend(backend, layers, slice(0, t0 + t))
        be_v = self._backend()
        be_v.model_runner = SimpleNamespace(decode_num_tokens_per_req=lambda: 6)
        layers_v = self._layers()
        self._run_extend(be_v, layers_v, slice(0, t0))
        ver = self._run_verify(be_v, layers_v, slice(t0, t0 + t))
        for lid in LAYER_ORDER:
            a = pref[lid][-t:]
            b = ver[lid]
            d = self._maxdiff(a, b)
            print(f"verify T=6 L{lid} maxabs={d:.5f}")
            self.assertTrue(
                torch.allclose(a.float(), b.float(), atol=ATOL, rtol=RTOL),
                f"verify T=6 L{lid} maxabs={d}",
            )

    def test_target_verify_cuda_graph_capture(self):
        t0, t = 64, 6
        fb = self._fb(False, target_verify=True)
        backend = self._backend()
        backend.model_runner = SimpleNamespace(decode_num_tokens_per_req=lambda: 6)
        layers = self._layers()
        self._run_extend(backend, layers, slice(0, t0))
        static_x = self.x[t0 : t0 + t].clone()
        static_ql = self.q_lora[t0 : t0 + t].clone()
        static_q = self.q[t0 : t0 + t].clone()
        static_pos = self.positions[t0 : t0 + t].clone()
        for lid in LAYER_ORDER:
            sm70_forward_low_ratio_sources(
                backend, layers[lid], static_x, static_ql, static_pos, fb
            )
            sm70_forward_sparse(
                backend, static_q, layers[lid], fb, layers[lid].compress_ratio, self.sink
            )
        ref = self._run_verify(backend, layers, slice(t0, t0 + t))
        st = get_state(backend)
        snap = {
            "swa": {k: v.clone() for k, v in st.swa_ring.items()},
            "kv": {k: v.clone() for k, v in st.kv_rows.items()},
            "ix": {k: v.clone() for k, v in st.index_rows.items()},
            "pkv": {k: v.clone() for k, v in st.pending_kv.items()},
            "psc": {k: v.clone() for k, v in st.pending_score.items()},
            "topk": {k: v.clone() for k, v in st.topk.items()},
            "vtopk": {k: v.clone() for k, v in st.verify_topk.items()},
            "cand": None if st.cand_ids is None else st.cand_ids.clone(),
            "vcand": None if st.verify_cand is None else st.verify_cand.clone(),
            "ck": None if st.cand_key_ids is None else st.cand_key_ids.clone(),
        }

        def restore():
            for k, v in snap["swa"].items():
                st.swa_ring[k].copy_(v)
            for k, v in snap["kv"].items():
                st.kv_rows[k].copy_(v)
            for k, v in snap["ix"].items():
                st.index_rows[k].copy_(v)
            for k, v in snap["pkv"].items():
                st.pending_kv[k].copy_(v)
            for k, v in snap["psc"].items():
                st.pending_score[k].copy_(v)
            for k, v in snap["topk"].items():
                st.topk[k].copy_(v)
            for k, v in snap["vtopk"].items():
                st.verify_topk[k].copy_(v)
            if snap["cand"] is not None:
                st.cand_ids.copy_(snap["cand"])
            if snap["vcand"] is not None:
                st.verify_cand.copy_(snap["vcand"])
            if snap["ck"] is not None:
                st.cand_key_ids.copy_(snap["ck"])

        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        captured = {}
        with torch.cuda.graph(g):
            for lid in LAYER_ORDER:
                sm70_forward_low_ratio_sources(
                    backend, layers[lid], static_x, static_ql, static_pos, fb
                )
                captured[lid] = sm70_forward_sparse(
                    backend,
                    static_q,
                    layers[lid],
                    fb,
                    layers[lid].compress_ratio,
                    self.sink,
                )
        restore()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        for lid in LAYER_ORDER:
            d = self._maxdiff(captured[lid], ref[lid])
            print(f"verify graph replay L{lid} maxabs={d:.5f}")
            self.assertTrue(
                torch.allclose(
                    captured[lid].float(), ref[lid].float(), atol=1e-3, rtol=1e-3
                ),
                f"verify graph L{lid} maxabs={d}",
            )


if __name__ == "__main__":
    unittest.main()
