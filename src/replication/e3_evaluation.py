"""Independent calibration and out-of-sample metrics for E3 Figure 5."""

from __future__ import annotations

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


def estimate_indifference_price(calibration_loss: np.ndarray, q: float, alpha: float) -> float:
    """Estimate p=q+CVaR_alpha(L_cal-q) from a fresh calibration sample."""
    losses = _loss_array(calibration_loss, "calibration_loss")
    if not np.isfinite(q):
        raise ValueError(f"q must be finite, got {q}")
    return indifference_price(float(q), torch.as_tensor(losses, dtype=torch.float64), alpha)


def summarize_oos(oos_loss: np.ndarray, price: float, alpha: float) -> dict[str, float]:
    """Summarize H=p-L on OOS paths; realized CVaR is the lower-tail CVaR of H."""
    losses = _loss_array(oos_loss, "oos_loss")
    if not np.isfinite(price):
        raise ValueError(f"price must be finite, got {price}")

    excess_loss = losses - float(price)
    excess_tensor = torch.as_tensor(excess_loss, dtype=torch.float64)
    hedging_error = -excess_loss
    return {
        # Paper Figure 5 does not define its "Mean Loss" statistic in the caption.
        # Use an explicitly labelled mean absolute hedging-error proxy instead.
        "mean_abs_hedging_error": float(np.mean(np.abs(hedging_error))),
        "price": float(price),
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
    """Price each policy on calibration paths and score their common OOS paths."""
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
