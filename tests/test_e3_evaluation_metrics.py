"""Unit tests for E3 independent-evaluation loss conventions."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.run_e3 import summarize


def test_e3_summary_separates_raw_and_centered_terminal_loss() -> None:
    calibration = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    oos = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = summarize(calibration, oos, alpha=0.8)
    price = result["indifference_price"]

    assert result["oos_mean_raw_loss"] == pytest.approx(float(oos.mean()))
    assert result["oos_signed_terminal_hedging_loss"] == pytest.approx(
        float((oos - price).mean())
    )
    assert result["oos_mean_hedging_error"] == pytest.approx(
        float((price - oos).mean())
    )
    assert result["oos_mean_hedging_error"] == pytest.approx(
        -result["oos_signed_terminal_hedging_loss"]
    )
    assert result["oos_mean_excess_loss_diagnostic"] == pytest.approx(
        float(np.maximum(oos - price, 0.0).mean())
    )
