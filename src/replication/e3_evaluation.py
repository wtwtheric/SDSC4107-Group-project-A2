"""Independent calibration and out-of-sample metrics for E3 Figure 5."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.common.metrics import cvar, indifference_price


def _loss_array(values: np.ndarray, name: str) -> np.ndarray:
    """Validate and return a finite one-dimensional terminal-loss array."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one observation")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite observations")
    return array


def estimate_indifference_price(
    calibration_loss: np.ndarray, q: float, alpha: float
) -> float:
    """Estimate p=q+CVaR_alpha(L_cal-q) from an independent calibration sample."""
    losses = _loss_array(calibration_loss, "calibration_loss")
    if not np.isfinite(q):
        raise ValueError(f"q must be finite, got {q}")
    tensor = torch.as_tensor(losses, dtype=torch.float64)
    return indifference_price(float(q), tensor, alpha)


def summarize_oos(
    oos_loss: np.ndarray, price: float, alpha: float
) -> dict[str, float]:
    """Summarize hedging error H=p-L on OOS paths using the upper-tail loss convention."""
    losses = _loss_array(oos_loss, "oos_loss")
    if not np.isfinite(price):
        raise ValueError(f"price must be finite, got {price}")

    excess_loss = losses - float(price)
    excess_tensor = torch.as_tensor(excess_loss, dtype=torch.float64)
    hedging_error = -excess_loss
    return {
        # The paper labels this column "Mean Loss" but does not define it in the caption.
        # We operationalize it as the expected positive shortfall beyond the charged price.
        "mean_loss": float(np.maximum(excess_loss, 0.0).mean()),
        "price": float(price),
        # Lower-tail CVaR of H = p - L; negative values indicate residual tail shortfall.
        "realized_cvar": float(-cvar(excess_tensor, alpha)),
        "mean_hedging_error": float(hedging_error.mean()),
        "oos_cvar_loss": float(cvar(torch.as_tensor(losses, dtype=torch.float64), alpha)),
    }


def build_e3_summary(
    calibration_loss_recurrent: np.ndarray,
    calibration_loss_simpler: np.ndarray,
    oos_loss_recurrent: np.ndarray,
    oos_loss_simpler: np.ndarray,
    *,
    q: float,
    alpha: float,
) -> dict[str, dict[str, float]]:
    """Price each policy on calibration paths and evaluate both on their common OOS paths."""
    calibration_r = _loss_array(calibration_loss_recurrent, "calibration_loss_recurrent")
    calibration_s = _loss_array(calibration_loss_simpler, "calibration_loss_simpler")
    oos_r = _loss_array(oos_loss_recurrent, "oos_loss_recurrent")
    oos_s = _loss_array(oos_loss_simpler, "oos_loss_simpler")
    if calibration_r.shape != calibration_s.shape:
        raise ValueError("recurrent and simpler calibration arrays must have the same shape")
    if oos_r.shape != oos_s.shape:
        raise ValueError("recurrent and simpler OOS arrays must have the same shape")

    price_r = estimate_indifference_price(calibration_r, q, alpha)
    price_s = estimate_indifference_price(calibration_s, q, alpha)
    return {
        "recurrent": summarize_oos(oos_r, price_r, alpha),
        "simpler": summarize_oos(oos_s, price_s, alpha),
    }


def make_evaluation_payload(
    *,
    source_metrics: dict[str, Any],
    calibration_seed: int,
    calibration_paths: int,
    oos_seed: int,
    alpha: float,
    q: float,
    summary: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Build the JSON-safe metadata payload for an independent E3 evaluation."""
    return {
        "schema_version": "1.0",
        "exp": "e3_independent_evaluation",
        "source": {
            "exp": source_metrics.get("exp"),
            "seed": int(source_metrics.get("seed", 0)),
            "git_commit": source_metrics.get("git_commit"),
            "config_sha256": source_metrics.get("config_sha256"),
        },
        "calibration": {
            "seed": int(calibration_seed),
            "n_paths": int(calibration_paths),
        },
        "oos": {
            "seed": int(oos_seed),
            "n_paths": int(source_metrics.get("n_paths", 0)),
        },
        "alpha": float(alpha),
        "q_zero_cost_reference": float(q),
        "definitions": {
            "hedging_error": "H = p_cal - L_oos",
            "price": "p_cal = q + CVaR_alpha(L_cal - q)",
            "mean_loss": "mean(max(L_oos - p_cal, 0))",
            "realized_cvar": "-CVaR_alpha(L_oos - p_cal), the lower-tail CVaR of H",
        },
        "metrics": summary,
    }
