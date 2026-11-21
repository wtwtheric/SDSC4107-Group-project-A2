"""Focused tests for E2cs support, training trend, and BN state diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scripts.a2.diagnose_e2cs_support_bn import (
    _batchnorm_diagnostics,
    support_metrics,
    training_diagnostics,
)
from src.replication.net_recurrent import StepMLP


def test_support_accounting_and_peak_neighborhoods() -> None:
    s = np.array([94.0, 99.8, 96.8, 104.0])
    v = np.array([0.04, 0.03, 0.03, 0.2])
    out = support_metrics(s, v, np.linspace(95, 103, 9), np.linspace(0.02, 0.14, 7))
    assert out["n_paths"] == 4
    assert out["fraction_inside_both_ranges"] == pytest.approx(0.5)
    assert out["fraction_in_model_peak_neighborhood"] == pytest.approx(0.25)
    assert out["fraction_in_shifted_peak_neighborhood"] == pytest.approx(0.25)


def test_bn_counterfactual_does_not_mutate_statistics() -> None:
    torch.manual_seed(7)
    step = StepMLP(n_in=2, n_hidden=17, n_out=2, norm="batch").double()
    step.train()
    with torch.no_grad():
        for _ in range(10):
            step(torch.randn(128, 2, dtype=torch.float64))
    step.eval()
    before = {key: value.clone() for key, value in step.state_dict().items()}
    x = torch.randn(200, 2, dtype=torch.float64)
    result = _batchnorm_diagnostics(step, x, torch.tensor([1.0, 100.0], dtype=torch.float64))
    assert result["bn1"]["num_batches_tracked"] == 10
    assert result["bn2"]["num_batches_tracked"] == 10
    assert result["bn1"]["standardized_mean_shift_median"] >= 0.0
    assert result["batch_statistics_counterfactual"]["rmse_change_stock"] >= 0.0
    for key, value in step.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_training_history_parses_each_alpha_separately(tmp_path) -> None:
    lines = []
    for alpha in (0.95, 0.99):
        lines.append(f"INFO training call-spread policy alpha={alpha:.2f}")
        for i in range(1, 13):
            lines.append(
                f"INFO iter {i * 2500}/30000 lr=0.001 "
                f"train_cvar=0.9 (last 2500 avg {1.0 + alpha - i * 0.02:.4f})"
            )
    path = tmp_path / "master.log"
    path.write_text("\n".join(lines), encoding="utf-8")
    result = training_diagnostics(path)
    assert len(result["0.95"]["series"]) == 12
    assert len(result["0.99"]["series"]) == 12
    assert result["0.99"]["final_window_mean"] > result["0.95"]["final_window_mean"]
    assert result["0.95"]["has_fixed_validation_history"] is False
