from __future__ import annotations

import json

import numpy as np
from scripts.run_e3 import (
    apply_evaluation_to_metrics,
    summarize,
    update_metrics_file,
)


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


def test_e3_status_distinguishes_execution_from_paper_targets(tmp_path) -> None:
    """Numerical mismatch must not turn a valid evaluation into a failed run."""
    report = {
        "calibration_paths": 1_000_000,
        "calibration_seed": 271828,
        "oos_paths": 1_000_000,
        "oos_seed": 314159,
        "device": "cuda",
        "runtime_sec": 20.0,
        "fixed_validation": {
            "paths": 10_000,
            "seed": 8675309,
            "metrics": {
                "recurrent": {"cvar": 4.199},
                "simpler": {"cvar": 4.257},
            },
        },
        "metrics": {
            "recurrent": {
                "indifference_price": 4.1707,
                "oos_realized_cvar_loss": 4.1666,
                "oos_realized_cvar_shortfall": -0.004,
                "oos_mean_raw_loss": 2.8704,
                "oos_signed_terminal_hedging_loss": -1.3003,
                "oos_mean_excess_loss_diagnostic": 0.000770,
            },
            "simpler": {
                "indifference_price": 4.2261,
                "oos_realized_cvar_loss": 4.2234,
                "oos_realized_cvar_shortfall": -0.0027,
                "oos_mean_raw_loss": 2.9323,
                "oos_signed_terminal_hedging_loss": -1.2938,
                "oos_mean_excess_loss_diagnostic": 0.000753,
            },
        },
    }
    original_checks = {
        "finite_prices": True,
        "nonnegative_transaction_costs": True,
        "common_oos_path_count": True,
        "recurrent_price_in_paper_band": False,
        "simpler_price_in_paper_band": False,
        "recurrent_price_lower_than_simpler": True,
    }
    metrics, checks = apply_evaluation_to_metrics({}, original_checks, report)
    assert metrics["mean_loss_recurrent"] == 0.000770
    assert metrics["signed_terminal_hedging_loss_recurrent"] == -1.3003
    assert "recurrent_price_in_paper_band" not in checks
    assert "recurrent_price_lower_than_simpler" not in checks
    assert checks["independent_validation_seed"] is True

    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps({
            "git_commit": "feb6c7b",
            "metrics": {"original": 123},
            "checks": original_checks,
            "passed": False,
        }), encoding="utf-8",
    )
    update_metrics_file(tmp_path, metrics, checks, report)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["git_commit"] == "feb6c7b"
    assert saved["metrics"]["original"] == 123
    assert saved["passed"] is True
    assert saved["operational_passed"] is True
    assert saved["paper_target_passed"] is False
    assert saved["paper_target_checks"]["recurrent_price_in_paper_band"] is False
    assert saved["economic_direction_checks"]["recurrent_price_lower_than_simpler"] is True
    assert saved["economic_direction_checks"][
        "recurrent_mean_positive_loss_proxy_lower_than_simpler"
    ] is False
    assert all(saved["checks"].values())
