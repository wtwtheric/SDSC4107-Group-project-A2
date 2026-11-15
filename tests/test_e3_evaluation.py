from __future__ import annotations

import numpy as np
import pytest
import torch
from src.common.metrics import cvar
from src.replication.e3_evaluation import (
    build_e3_summary,
    estimate_indifference_price,
    summarize_oos,
)


def test_indifference_price_uses_calibration_distribution() -> None:
    calibration = np.array([0.0, 1.0, 2.0, 3.0, 10.0], dtype=np.float64)
    q = 1.69
    expected = float(cvar(torch.from_numpy(calibration), 0.8))
    assert estimate_indifference_price(calibration, q, 0.8) == pytest.approx(expected)


def test_oos_realized_cvar_is_lower_tail_of_p_minus_loss() -> None:
    losses = np.array([0.0, 1.0, 2.0, 3.0, 10.0], dtype=np.float64)
    price = 3.0
    result = summarize_oos(losses, price, 0.8)
    hedging_error = price - losses
    expected = float(torch.mean(torch.tensor([-7.0])))  # worst 20% is the single minimum
    assert result["realized_cvar"] == pytest.approx(expected)
    assert result["mean_abs_hedging_error"] == pytest.approx(float(np.mean(np.abs(hedging_error))))


def test_e3_summary_prices_on_calibration_and_scores_common_oos() -> None:
    calibration_r = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    calibration_s = np.array([0.0, 1.0, 2.0, 4.0, 12.0])
    oos_r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    oos_s = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    result = build_e3_summary(calibration_r, calibration_s, oos_r, oos_s, q=1.69, alpha=0.8)
    assert result["recurrent"]["price"] == pytest.approx(10.0)
    assert result["simpler"]["price"] == pytest.approx(12.0)
    assert result["recurrent"]["realized_cvar"] == pytest.approx(5.0)
    assert result["simpler"]["realized_cvar"] == pytest.approx(6.5)


@pytest.mark.parametrize(
    "values",
    [
        np.array([], dtype=np.float64),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
    ],
)
def test_calibration_rejects_empty_or_non_finite_losses(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        estimate_indifference_price(values, 1.69, 0.99)


def test_summary_rejects_unpaired_oos_lengths() -> None:
    with pytest.raises(ValueError, match="OOS arrays"):
        build_e3_summary(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([1.0]),
            q=1.69,
            alpha=0.5,
        )
