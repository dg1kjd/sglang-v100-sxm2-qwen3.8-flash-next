"""CPU tests for SM70 CSA2 ring/pending checkpoints."""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.dsv4.sm70_csa2_boundary import (
    cap_verify_commit,
    csa2_finish_forward,
    csa2_prepare_decode,
    csa2_prepare_extend,
    evict_recent,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _State:
    def __init__(self) -> None:
        self.swa_ring = {0: torch.zeros(4, dtype=torch.uint8)}
        self.pending_kv = {1: torch.zeros(2, dtype=torch.float32)}
        self.pending_score = {1: torch.zeros(2, dtype=torch.float32)}


class _Backend:
    def __init__(self, state=None) -> None:
        if state is not None:
            self._sm70_csa2 = state


class TestCsa2Boundary(CustomTestCase):
    def test_cap_verify_commit_drops_the_overshoot(self):
        # Live failure: prompt 65, max_new_tokens 8, pin 73, uncapped image 78.
        self.assertEqual(cap_verify_commit(71, 6, 73), 2)
        self.assertEqual(71 + cap_verify_commit(71, 6, 73), 73)
        self.assertEqual(cap_verify_commit(65, 6, 73), 6)
        self.assertEqual(cap_verify_commit(78, 6, 73), 0)
        self.assertEqual(cap_verify_commit(71, 6, None), 6)

    def test_evict_recent_keeps_the_tail(self):
        self.assertEqual(evict_recent([1, 2, 3], 2), [2, 3])
        self.assertEqual(evict_recent([1], 8), [1])

    def test_rewind_restores_ring_and_drops_the_tail(self):
        target = _Backend(_State())
        backends = [target]
        csa2_prepare_extend(backends, 0)
        target._sm70_csa2.swa_ring[0].fill_(1)
        target._sm70_csa2.pending_kv[1].fill_(1)
        csa2_finish_forward(backends, 4, from_extend=True)
        csa2_prepare_decode(backends)
        store = target._csa2_boundary
        self.assertIn(4, store.history)
        self.assertEqual(store.resident_end, 4)

        target._sm70_csa2.swa_ring[0].fill_(7)
        target._sm70_csa2.pending_kv[1].fill_(7)
        csa2_finish_forward(backends, 6, from_extend=False)
        csa2_prepare_extend(backends, 4)

        self.assertTrue(
            torch.equal(
                target._sm70_csa2.swa_ring[0], torch.ones(4, dtype=torch.uint8)
            )
        )
        self.assertTrue(
            torch.equal(target._sm70_csa2.pending_kv[1], torch.ones(2))
        )
        self.assertEqual(store.resident_end, 4)
        self.assertNotIn(6, store.history)
        self.assertIn(4, store.order)

    def test_prefix_zero_drops_the_resident_image(self):
        target = _Backend(_State())
        backends = [target]
        target._sm70_csa2.swa_ring[0].fill_(3)
        csa2_finish_forward(backends, 4, from_extend=True)
        csa2_prepare_decode(backends)
        self.assertIn(4, target._csa2_boundary.history)
        csa2_prepare_extend(backends, 0)
        self.assertEqual(target._csa2_boundary.history, {})
        self.assertEqual(target._csa2_boundary.resident_end, 0)
        self.assertEqual(int(target._sm70_csa2.swa_ring[0].sum()), 12)

    def test_missing_length_raises(self):
        target = _Backend(_State())
        backends = [target]
        csa2_finish_forward(backends, 4, from_extend=True)
        csa2_prepare_decode(backends)
        with self.assertRaises(RuntimeError):
            csa2_prepare_extend(backends, 2)

    def test_bonus_snap_is_filed_under_the_pin(self):
        # Scheduler pin 16, worker tip 17. The bytes stay, under the pin.
        target = _Backend(_State())
        backends = [target]
        target._sm70_csa2.swa_ring[0].fill_(1)
        csa2_finish_forward(backends, 5, from_extend=True)
        csa2_prepare_decode(backends)
        target._sm70_csa2.swa_ring[0].fill_(9)
        csa2_finish_forward(backends, 17, from_extend=False)
        csa2_prepare_extend(backends, 16)
        self.assertEqual(int(target._sm70_csa2.swa_ring[0][0]), 9)
        self.assertEqual(target._csa2_boundary.resident_end, 16)
        self.assertIn(16, target._csa2_boundary.history)
        self.assertNotIn(17, target._csa2_boundary.history)

    def test_rewind_to_bonus_pin_after_a_later_turn(self):
        # Live failure: pin 23995 filed as 23996, then the tip moved to 24717.
        target = _Backend(_State())
        backends = [target]
        target._sm70_csa2.swa_ring[0].fill_(1)
        csa2_finish_forward(backends, 5, from_extend=True)
        csa2_prepare_decode(backends)
        target._sm70_csa2.swa_ring[0].fill_(9)
        csa2_finish_forward(backends, 17, from_extend=False)
        csa2_prepare_extend(backends, 16)
        target._sm70_csa2.swa_ring[0].fill_(3)
        csa2_finish_forward(backends, 20, from_extend=False)
        csa2_prepare_extend(backends, 16)
        self.assertEqual(int(target._sm70_csa2.swa_ring[0][0]), 9)
        self.assertEqual(target._csa2_boundary.resident_end, 16)
        self.assertNotIn(20, target._csa2_boundary.history)
        self.assertNotIn(17, target._csa2_boundary.history)

    def test_verify_gap_of_two_is_filed_under_the_pin(self):
        # Third turn: pin 23696, verify tip 23698. Nothing snapshotted between.
        target = _Backend(_State())
        backends = [target]
        target._sm70_csa2.swa_ring[0].fill_(1)
        csa2_finish_forward(backends, 5, from_extend=True)
        csa2_prepare_decode(backends)
        target._sm70_csa2.swa_ring[0].fill_(9)
        csa2_finish_forward(backends, 18, from_extend=False)
        csa2_prepare_extend(backends, 16)
        self.assertEqual(int(target._sm70_csa2.swa_ring[0][0]), 9)
        self.assertEqual(target._csa2_boundary.resident_end, 16)
        self.assertIn(16, target._csa2_boundary.history)
        self.assertNotIn(18, target._csa2_boundary.history)

    def test_verify_slack_stops_at_an_intermediate_snap(self):
        target = _Backend(_State())
        backends = [target]
        target._sm70_csa2.swa_ring[0].fill_(1)
        csa2_finish_forward(backends, 16, from_extend=True)
        csa2_prepare_decode(backends)
        target._sm70_csa2.swa_ring[0].fill_(4)
        csa2_finish_forward(backends, 20, from_extend=False)
        with self.assertRaises(RuntimeError):
            csa2_prepare_extend(backends, 10)
        self.assertNotIn(10, target._csa2_boundary.history)
        self.assertIn(20, target._csa2_boundary.history)
        self.assertEqual(int(target._sm70_csa2.swa_ring[0][0]), 4)

    def test_extend_stop_one_past_the_pin_is_not_relabeled(self):
        target = _Backend(_State())
        backends = [target]
        target._sm70_csa2.swa_ring[0].fill_(1)
        csa2_finish_forward(backends, 4, from_extend=True)
        csa2_prepare_decode(backends)
        target._sm70_csa2.swa_ring[0].fill_(6)
        csa2_finish_forward(backends, 6, from_extend=True)
        with self.assertRaises(RuntimeError):
            csa2_prepare_extend(backends, 5)
        self.assertEqual(int(target._sm70_csa2.swa_ring[0][0]), 6)
        self.assertIn(6, target._csa2_boundary.history)
        self.assertNotIn(5, target._csa2_boundary.history)

    def test_draft_ring_rewinds_with_the_target(self):
        target = _Backend(_State())
        draft = _Backend(_State())
        backends = [target, draft]
        target._sm70_csa2.swa_ring[0].fill_(1)
        draft._sm70_csa2.swa_ring[0].fill_(2)
        csa2_finish_forward(backends, 4, from_extend=True)
        csa2_prepare_decode(backends)
        target._sm70_csa2.swa_ring[0].fill_(8)
        draft._sm70_csa2.swa_ring[0].fill_(9)
        csa2_finish_forward(backends, 6, from_extend=False)
        csa2_prepare_extend(backends, 4)
        self.assertEqual(int(target._sm70_csa2.swa_ring[0][0]), 1)
        self.assertEqual(int(draft._sm70_csa2.swa_ring[0][0]), 2)
