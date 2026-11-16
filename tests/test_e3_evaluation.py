from __future__ import annotations

import json

import numpy as np
from scripts.run_e3 import summarize, update_metrics_file


def test_summarize_uses_independent_calibration_price_and_oos_shortfall() -> None:
    calibration = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    oos = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    result = summarize(calibration, oos, alpha=0.8)

    assert result["indifference_price"] == 4.0
    assert result["oos_mean_raw_loss"] == 2.0
    assert result["oos_signed_terminal_hedging_loss"] == -2.0
    assert result["oos_mean_hedging_error"] == 2.0
    assert result["oos_realized_cvar_shortfall"] == 0.0


def test_summarize_does_not_use_oos_sample_to_set_price() -> None:
    calibration = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    oos = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    result = summarize(calibration, oos, alpha=0.8)

    assert result["indifference_price"] == 5.0
    assert result["oos_realized_cvar_shortfall"] == 9.0


def test_eval_only_metadata_uses_nested_fixed_validation(tmp_path) -> None:
    """Regression: report has fixed_validation, not top-level validation_paths."""
    metrics_path = tmp_path / "metrics.json"
    original = {
        "seed": 0,
        "git_commit": "original-training-commit",
        "metrics": {"original_training_metric": 12.5},
        "checks": {"finite_prices": True},
        "passed": True,
    }
    metrics_path.write_text(json.dumps(original), encoding="utf-8")
    report = {
        "calibration_paths": 1_000_000,
        "calibration_seed": 271828,
        "oos_paths": 1_000_000,
        "oos_seed": 314159,
        "device": "cuda",
        "runtime_sec": 2.5,
        "fixed_validation": {
            "paths": 10_000,
            "seed": 8675309,
            "metrics": {"recurrent": {"cvar": 4.1}, "simpler": {"cvar": 4.2}},
        },
    }

    update_metrics_file(
        tmp_path, {"price_recurrent": 4.17}, {"independent_seeds": True}, report
    )

    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved["seed"] == 0
    assert saved["git_commit"] == "original-training-commit"
    assert saved["metrics"]["original_training_metric"] == 12.5
    assert saved["metrics"]["price_recurrent"] == 4.17
    assert saved["passed"] is True
    assert saved["independent_evaluation"]["validation_paths"] == 10_000
    assert saved["independent_evaluation"]["validation_seed"] == 8675309
    assert saved["independent_evaluation"]["validation_metrics"]["simpler"]["cvar"] == 4.2
