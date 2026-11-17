"""Diagnose corrected E2CS simpler policies without retraining or 1M-path OOS simulation.

Read the archived seed-0 checkpoints (two alpha policies), rebuild the same
(S,V) grid used in Figures 8-9, compare both stock and variance-swap hedge
positions with the Heston model hedge, and save auditable shape statistics.

Examples:
    python -m scripts.diagnose_e2cs
    python -m scripts.diagnose_e2cs --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.make_e2cs_figures import plot_surface_pairs
from src.common.config import load_config
from src.common.seed import resolve_device
from src.replication.e1 import market_from_cfg, policy_from_cfg
from src.replication.e2cs import _model_delta_grids


def _policy_grids(
    policy: torch.nn.Module,
    spots: np.ndarray,
    variances: np.ndarray,
    step: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Direct feed-forward F_k(S,V) on a fixed grid; BatchNorm stays in eval mode."""
    spot, variance = np.meshgrid(spots, variances, indexing="ij")
    inputs = torch.as_tensor(
        np.column_stack((spot.ravel(), variance.ravel())),
        dtype=torch.float64,
        device=device,
    )
    policy.eval()
    with torch.no_grad():
        position = policy.steps[step](inputs) * policy.out_scale
    value = position.detach().cpu().numpy().reshape(len(spots), len(variances), 2)
    if not np.isfinite(value).all():
        raise ValueError("Non-finite simpler policy output on the diagnostic grid")
    return value[..., 0], value[..., 1]


def _metrics(
    model: np.ndarray,
    policy: np.ndarray,
    spots: np.ndarray,
    variances: np.ndarray,
) -> dict[str, float | None]:
    if model.shape != policy.shape or model.shape != (len(spots), len(variances)):
        raise ValueError("Grid and position shapes differ")
    if not (np.isfinite(model).all() and np.isfinite(policy).all()):
        raise ValueError("Non-finite hedge values")
    difference = policy - model
    model_centered = model - model.mean()
    policy_centered = policy - policy.mean()
    covariance_denom = float(np.sqrt(np.sum(model_centered**2) * np.sum(policy_centered**2)))
    correlation = (
        float(np.sum(model_centered * policy_centered) / covariance_denom)
        if covariance_denom > 1e-14 else None
    )
    mi = np.unravel_index(int(np.argmax(model)), model.shape)
    ni = np.unravel_index(int(np.argmax(policy)), policy.shape)
    mid = len(variances) // 2
    low_variance = 0
    return {
        "model_min": float(model.min()),
        "model_max": float(model.max()),
        "model_range": float(np.ptp(model)),
        "policy_min": float(policy.min()),
        "policy_max": float(policy.max()),
        "policy_range": float(np.ptp(policy)),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mae": float(np.mean(np.abs(difference))),
        "mean_signed_difference": float(difference.mean()),
        "max_abs_difference": float(np.max(np.abs(difference))),
        "correlation": correlation,
        "model_peak_spot": float(spots[mi[0]]),
        "model_peak_variance": float(variances[mi[1]]),
        "model_peak_height": float(model[mi]),
        "policy_peak_spot": float(spots[ni[0]]),
        "policy_peak_variance": float(variances[ni[1]]),
        "policy_peak_height": float(policy[ni]),
        "policy_at_model_peak": float(policy[mi]),
        "difference_at_model_peak": float(difference[mi]),
        "fraction_policy_negative": float(np.mean(policy < 0.0)),
        "fraction_policy_below_model": float(np.mean(difference < 0.0)),
        "low_variance_policy_spot_range": float(np.ptp(policy[:, low_variance])),
        "low_variance_model_spot_range": float(np.ptp(model[:, low_variance])),
        "mid_variance_policy_spot_range": float(np.ptp(policy[:, mid])),
        "mid_variance_model_spot_range": float(np.ptp(model[:, mid])),
    }


def analyze(
    *, checkpoint_root: Path, config_path: Path, output_dir: Path,
    figures_dir: Path, seed: int, step: int, device: torch.device,
    make_plots: bool = True,
) -> dict[str, Any]:
    cfg = load_config(str(config_path))
    market = market_from_cfg(cfg)
    if market.d != 2 or float(cfg.get("eps", 0.0)) != 0.0:
        raise ValueError("E2cs diagnosis requires Heston d=2 and no transaction costs")
    if not (0 <= step < market.n_steps):
        raise ValueError(f"Invalid hedge step {step}")
    spots = np.linspace(95.0, 103.0, 81)
    variances = np.linspace(0.02, 0.14, 61)
    strike1 = float(cfg["payoff"]["strike1"])
    strike2 = float(cfg["payoff"]["strike2"])
    model_stock, model_varswap = _model_delta_grids(
        market, strike1, strike2, step, spots, variances,
    )
    outputs: dict[str, dict[str, np.ndarray]] = {}
    metrics_by_alpha: dict[str, Any] = {}
    for alpha in ("0.95", "0.99"):
        path = checkpoint_root / alpha / str(seed)
        saved = load_config(str(path / "config.yaml"))
        if (
            saved["policy"].get("recurrent", True)
            or saved["policy"].get("norm") != "batch"
            or saved["payoff"]["strike1"] != strike1
            or saved["payoff"]["strike2"] != strike2
        ):
            raise ValueError(f"{path}: checkpoint is not the corrected simpler E2cs setup")
        policy = policy_from_cfg(
            saved, market.n_features, market.d, device,
            recurrent=False, n_features_override=2,
        )
        policy.load_state_dict(
            torch.load(path / "policy.pt", map_location=device, weights_only=True),
            strict=True,
        )
        stock, varswap = _policy_grids(policy, spots, variances, step, device)
        outputs[alpha] = {
            "model_spot_grid": spots,
            "model_variance_grid": variances,
            "model_stock_delta_grid": model_stock,
            "policy_stock_delta_grid": stock,
            "model_varswap_delta_grid": model_varswap,
            "policy_varswap_delta_grid": varswap,
        }
        metrics_by_alpha[alpha] = {
            "stock": _metrics(model_stock, stock, spots, variances),
            "variance_swap": _metrics(model_varswap, varswap, spots, variances),
            "checkpoint": str(path / "policy.pt"),
            "config": str(path / "config.yaml"),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "seed": seed,
        "step": step,
        "spot_grid": [float(spots[0]), float(spots[-1]), len(spots)],
        "variance_grid": [float(variances[0]), float(variances[-1]), len(variances)],
        "architecture": "simpler, per-date F_k(S,V), batch normalization, two hedging legs",
        "notes": (
            "Uniform-grid diagnostics, not a probability-weighted OOS evaluation. "
            "Policy surfaces are evaluated directly from the saved checkpoints; "
            "model surfaces use the existing finite-difference Heston pricer."
        ),
        "alpha": metrics_by_alpha,
    }
    (output_dir / "e2cs_shape_diagnostics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    with (output_dir / "e2cs_shape_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        names = [
            "alpha", "leg", "rmse", "mae", "correlation", "model_peak_spot",
            "model_peak_height", "policy_peak_spot", "policy_peak_height",
            "policy_at_model_peak", "difference_at_model_peak",
            "model_range", "policy_range", "fraction_policy_below_model",
            "low_variance_model_spot_range", "low_variance_policy_spot_range",
        ]
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for alpha, d in metrics_by_alpha.items():
            for leg in ("stock", "variance_swap"):
                writer.writerow({
                    "alpha": alpha, "leg": leg,
                    **{key: d[leg][key] for key in names if key not in {"alpha", "leg"}},
                })
    np.savez_compressed(
        output_dir / "e2cs_diagnostic_grids.npz",
        spot_grid=spots,
        variance_grid=variances,
        model_stock=model_stock,
        model_varswap=model_varswap,
        policy_stock_95=outputs["0.95"]["policy_stock_delta_grid"],
        policy_stock_99=outputs["0.99"]["policy_stock_delta_grid"],
        policy_varswap_95=outputs["0.95"]["policy_varswap_delta_grid"],
        policy_varswap_99=outputs["0.99"]["policy_varswap_delta_grid"],
    )
    if make_plots:
        plot_surface_pairs(outputs, figures_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root", type=Path,
        default=Path("backups/e2cs_simple_seed0/e2cs"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/exp/e2cs.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/e2cs"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/e2cs_diagnostics"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = analyze(
        checkpoint_root=args.checkpoint_root,
        config_path=args.config,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        seed=args.seed,
        step=args.step,
        device=resolve_device(args.device),
        make_plots=not args.no_plots,
    )
    for alpha in ("0.95", "0.99"):
        s = result["alpha"][alpha]["stock"]
        print(
            f"alpha={alpha}: stock RMSE={s['rmse']:.4f}, "
            f"peak model={s['model_peak_height']:.4f}, "
            f"policy={s['policy_peak_height']:.4f}, "
            f"policy at model peak={s['policy_at_model_peak']:.4f}"
        )
    print(f"saved diagnostics under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
