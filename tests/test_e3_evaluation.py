from __future__ import annotations

import numpy as np

from scripts.run_e3 import summarize


def test_summarize_uses_independent_calibration_price_and_oos_shortfall() -> None:
    calibration = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    oos = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    result = summarize(calibration, oos, alpha=0.8)

    assert result["indifference_price"] == 4.0
    assert result["oos_mean_loss"] == 2.0
    assert result["oos_mean_hedging_error"] == 2.0
    assert result["oos_realized_cvar_shortfall"] == 0.0


def test_summarize_does_not_use_oos_sample_to_set_price() -> None:
    calibration = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    oos = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    result = summarize(calibration, oos, alpha=0.8)

    assert result["indifference_price"] == 5.0
    assert result["oos_realized_cvar_shortfall"] == 9.0
