"""Tests for post-training BatchNorm recalibration."""

from __future__ import annotations

import pytest
import torch

from src.replication.net_recurrent import SemiRecurrentPolicy
from src.replication.bn_recalibration import recalibrate_batchnorm_from_features


def _policy() -> SemiRecurrentPolicy:
    torch.manual_seed(7)
    return SemiRecurrentPolicy(
        n_features=2,
        d=2,
        n_steps=3,
        hidden=17,
        recurrent=False,
        norm="batch",
        out_scale=[1.0, 100.0],
    ).double()


def test_recalibration_changes_only_bn_running_moments() -> None:
    policy = _policy()
    before = {name: value.clone() for name, value in policy.state_dict().items()}
    torch.manual_seed(11)
    features = torch.randn(4096, 3, 2, dtype=torch.float64)
    features[..., 0] = 100.0 + 4.0 * features[..., 0]
    features[..., 1] = torch.clamp(0.04 + 0.08 * features[..., 1], min=0.0)

    report = recalibrate_batchnorm_from_features(
        policy, features, batch_size=777, device="cpu"
    )
    after = policy.state_dict()
    assert report["only_bn_running_moments_changed"]
    assert len(report["changed_buffers"]) == 3 * 4
    for name, value in after.items():
        if name.endswith(("running_mean", "running_var")):
            continue
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for step in range(3):
        assert int(policy.steps[step].bn1.num_batches_tracked) == 0
        assert int(policy.steps[step].bn2.num_batches_tracked) == 0
        assert (
            report["step_diagnostics"][step]["bn1_after_recalibration"][
                "standardized_mean_shift_max"
            ]
            < 1e-10
        )
        assert (
            report["step_diagnostics"][step]["bn2_after_recalibration"][
                "standardized_mean_shift_max"
            ]
            < 1e-10
        )


def test_recalibration_is_independent_of_chunk_size() -> None:
    torch.manual_seed(17)
    features = torch.randn(1200, 3, 2, dtype=torch.float64)
    features[..., 0] = 100.0 + 2.0 * features[..., 0]
    features[..., 1] = torch.clamp(0.04 + 0.03 * features[..., 1], min=0.0)
    first = _policy()
    second = _policy()
    recalibrate_batchnorm_from_features(first, features, batch_size=1200)
    recalibrate_batchnorm_from_features(second, features, batch_size=113)
    for (name_a, value_a), (name_b, value_b) in zip(
        first.state_dict().items(), second.state_dict().items(), strict=True
    ):
        assert name_a == name_b
        torch.testing.assert_close(value_a, value_b, rtol=1e-12, atol=1e-12)


def test_recalibration_rejects_recurrent_and_wrong_feature_shapes() -> None:
    policy = _policy()
    with pytest.raises(ValueError):
        recalibrate_batchnorm_from_features(
            policy, torch.randn(100, 3, 3, dtype=torch.float64)
        )
