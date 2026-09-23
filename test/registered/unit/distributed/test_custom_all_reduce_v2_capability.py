from unittest.mock import Mock

import pytest

from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo
from sglang.srt.distributed.device_communicators import custom_all_reduce_v2
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    AllReduceConfig,
    CustomAllReduceV2,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _patch_group(monkeypatch, *, world_size, same_node):
    group = object()
    device = object()
    monkeypatch.setattr(
        custom_all_reduce_v2.dist,
        "get_world_size",
        lambda group: world_size,
    )
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "get_supported_world_sizes",
        lambda: (world_size,),
    )
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "in_the_same_node_as",
        lambda group, source_rank: [same_node] * world_size,
    )
    return group, device


@pytest.mark.parametrize(
    ("same_node", "has_fabric_clique", "uses_vmm", "expected"),
    [
        (False, True, True, True),
        (False, False, True, False),
        (False, True, False, False),
        (True, None, None, True),
    ],
)
def test_topology_capability(
    monkeypatch, same_node, has_fabric_clique, uses_vmm, expected
):
    world_size = 8 if same_node else 16
    group, device = _patch_group(
        monkeypatch,
        world_size=world_size,
        same_node=same_node,
    )

    def is_one_clique(group, device):
        if same_node:
            pytest.fail("intra-node groups do not need a fabric clique")
        return has_fabric_clique

    def is_vmm_backed(device):
        if same_node:
            pytest.fail("intra-node groups do not need VMM")
        return uses_vmm

    intra_node_capability = Mock(return_value=True)

    monkeypatch.setattr(
        custom_all_reduce_v2,
        "is_one_nvlink_clique",
        is_one_clique,
    )
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "_is_vmm_backed_allocator",
        is_vmm_backed,
    )
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "can_use_custom_all_reduce_with_nvlink",
        intra_node_capability,
    )

    assert custom_all_reduce_v2.can_use_custom_all_reduce_v2(group, device) is expected
    if same_node:
        intra_node_capability.assert_called_once_with(
            group=group,
            device=device,
            supported_world_size=[world_size],
            cls_name="CustomAllReduceV2",
        )
    else:
        intra_node_capability.assert_not_called()


def test_pcie_opt_in_default_off_rejects_non_nvlink(monkeypatch):
    group, device = _patch_group(monkeypatch, world_size=4, same_node=True)
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "can_use_custom_all_reduce_with_nvlink",
        Mock(return_value=False),
    )
    with envs.SGLANG_CUSTOM_AR_ALLOW_PCIE.override(False):
        assert custom_all_reduce_v2.can_use_custom_all_reduce_v2(group, device) is False


def test_pcie_opt_in_allows_non_nvlink_intra_node(monkeypatch):
    group, device = _patch_group(monkeypatch, world_size=4, same_node=True)
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "can_use_custom_all_reduce_with_nvlink",
        Mock(return_value=False),
    )
    with envs.SGLANG_CUSTOM_AR_ALLOW_PCIE.override(True):
        assert custom_all_reduce_v2.can_use_custom_all_reduce_v2(group, device) is True


def test_pcie_opt_in_still_rejects_when_nvlink_probe_fails(monkeypatch):
    group, device = _patch_group(monkeypatch, world_size=4, same_node=True)
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "can_use_custom_all_reduce_with_nvlink",
        Mock(return_value=None),
    )
    with envs.SGLANG_CUSTOM_AR_ALLOW_PCIE.override(True):
        assert custom_all_reduce_v2.can_use_custom_all_reduce_v2(group, device) is False


def test_pcie_only_pick_config_is_push_and_size_capped():
    comm = CustomAllReduceV2.__new__(CustomAllReduceV2)
    comm.disabled = True
    comm.pcie_only = True
    comm.max_push_size = 128 * 1024
    assert comm._pick_config(4 * 1024, can_use_graph=True) == AllReduceConfig(
        AllReduceAlgo.ONE_SHOT_PUSH
    )
    assert comm._pick_config(128 * 1024, can_use_graph=False) == AllReduceConfig(
        AllReduceAlgo.ONE_SHOT_PUSH
    )
    assert comm._pick_config(128 * 1024 + 16, can_use_graph=True) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
