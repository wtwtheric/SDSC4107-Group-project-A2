"""Tests for the original-NPZ, frozen-BN E2cs sensitivity audit."""

from __future__ import annotations

import numpy as np
import torch

from scripts.diagnose_e2cs_original_bn import _grid_error, frozen_and_batch_moments
from src.replication.net_recurrent import StepMLP


def test_recompute_bn_moments_never_mutates_checkpoint() -> None:
    torch.manual_seed(42)
    net = StepMLP(n_in=2, n_hidden=17, n_out=2, norm="batch").double()
    net.train()
    with torch.no_grad():
        for _ in range(10):
            net(torch.randn((256, 2), dtype=torch.float64))
    net.eval()
    state = torch.tensor(
        [[95.0 + i * 0.03, 0.0 if i % 4 else 0.04] for i in range(256)],
        dtype=torch.float64,
    )
    initial = {key: value.clone() for key, value in net.state_dict().items()}
    frozen, alternative, stats = frozen_and_batch_moments(
        net, state, torch.tensor([1.0, 100.0], dtype=torch.float64)
    )
    with torch.no_grad():
        ref = net(state) * torch.tensor([1.0, 100.0], dtype=torch.float64)
    np.testing.assert_allclose(frozen, ref.numpy(), atol=1e-10)
    assert frozen.shape == alternative.shape == (256, 2)
    assert stats["bn1"]["num_batches_tracked"] == 10
    assert stats["bn2"]["num_batches_tracked"] == 10
    for key, value in net.state_dict().items():
        torch.testing.assert_close(value, initial[key])


def test_grid_error_conditions_on_inside_states() -> None:
    s = np.array([95.5, 96.0, 100.0, 110.0])
    v = np.array([0.02, 0.04, 0.04, 0.1])
    stock_grid = np.ones((3, 2)) * 0.4
    out = _grid_error(
        s, v, np.array([0.5, 0.3, 0.6, 0.0]),
        np.array([95.0, 97.5, 100.0]), np.array([0.02, 0.04]),
        stock_grid,
    )
    assert out["n_inside"] == 3
    assert out["fraction_inside"] == 0.75
    assert abs(out["rmse"] - np.sqrt((0.1**2 + (-0.1)**2 + 0.2**2) / 3)) < 1e-12
