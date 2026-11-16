"""Independent E3 calibration and Figure 5 evaluation from saved policies.

This script does NOT retrain either policy. It estimates each policy's
indifference price on fresh calibration paths, then evaluates centered
hedging errors on the independent OOS losses saved by the original E3 run.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.common.metrics import cvar
from src.common.seed import resolve_device, seed_everything
from src.replication.e1 import market_from_cfg, policy_from_cfg
from src.replication.env_heston import HestonWorld


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return cfg


def load_policy(
    path: Path, cfg: dict[str, Any], recurrent: bool, world: HestonWorld,
    device: torch.device,
) -> torch.nn.Module:
    policy = policy_from_cfg(
        cfg, world.market.n_features, world.market.d, device, recurrent=recurrent
    )
    state = torch.load(path, map_location=device, weights_only=True)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy


@torch.no_grad()
def calibration_losses(
    world: HestonWorld,
    recurrent: torch.nn.Module,
    simpler: torch.nn.Module,
    n_paths: int,
    batch_size: int,
    seed: int,
    log: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate both frozen policies on exactly the same fresh calibration paths."""
    seed_everything(seed)
    r_parts: list[np.ndarray] = []
    s_parts: list[np.ndarray] = []
    done = 0
    while done < n_paths:
        n = min(batch_size, n_paths - done)
        paths = world.simulate(n)
        features = world.features(paths)
        r_pos = recurrent.forward_path(features)
        s_pos = simpler.forward_path(features)
        r_parts.append(world.terminal_loss(r_pos, paths).cpu().numpy())
        s_parts.append(world.terminal_loss(s_pos, paths).cpu().numpy())
        done += n
        if done % max(batch_size * 4, batch_size) == 0 or done == n_paths:
            log.info("calibration paths %d/%d", done, n_paths)
    return np.concatenate(r_parts), np.concatenate(s_parts)


def empirical_cvar(values: np.ndarray, alpha: float) -> float:
    return float(cvar(torch.as_tensor(values, dtype=torch.float64), alpha))


def summarize(
    name: str, calibration: np.ndarray, oos_loss: np.ndarray, alpha: float
) -> dict[str, float]:
    # CVaR is translation equivariant: p = CVaR_alpha(L_cal).
    price = empirical_cvar(calibration, alpha)
    # Figure 5 plots p - Z + hedge P&L - transaction costs = p - terminal_loss.
    hedging_error = price - oos_loss
    # Positive shortfall means the OOS upper-tail loss exceeds the calibrated price.
    centered_loss = oos_loss - price
    return {
        "calibration_mean_loss": float(calibration.mean()),
        "calibration_cvar": price,
        "indifference_price": price,
        "oos_mean_loss": float(oos_loss.mean()),
        "oos_mean_hedging_error": float(hedging_error.mean()),
        "oos_realized_cvar_shortfall": empirical_cvar(centered_loss, alpha),
        "oos_realized_cvar_loss": empirical_cvar(oos_loss, alpha),
        "oos_hedging_error_q01": float(np.quantile(hedging_error, 0.01)),
        "oos_hedging_error_median": float(np.median(hedging_error)),
        "oos_hedging_error_q99": float(np.quantile(hedging_error, 0.99)),
    }


def make_figure(
    recurrent_error: np.ndarray,
    simpler_error: np.ndarray,
    metrics: dict[str, dict[str, float]],
    alpha: float,
    output: Path,
    bins: int,
) -> None:
    """Create a paper-style overlaid Figure 5 histogram with a metrics table."""
    pooled = np.concatenate([recurrent_error, simpler_error])
    lo, hi = np.quantile(pooled, [0.001, 0.999])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("Cannot construct histogram range from hedging errors")
    edges = np.linspace(lo, hi, bins + 1)

    fig = plt.figure(figsize=(11.5, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.5, 1.45])
    ax = fig.add_subplot(gs[0])
    ax.hist(recurrent_error, bins=edges, alpha=0.62, label="recurrent",
            color="#4472C4", edgecolor="none")
    ax.hist(simpler_error, bins=edges, alpha=0.52, label="simpler",
            color="#ED7D31", edgecolor="none")
    ax.axvline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.set_title("E3: Recurrent vs simpler deep hedging (independent calibration and OOS)")
    ax.set_xlabel(r"Terminal hedging error  $p_0 - L_T$")
    ax.set_ylabel("Number of OOS paths")
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.2)

    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")
    columns = ["Model", "Mean OOS loss", "Indifference price",
               f"Realized CVaR{alpha:.2f} shortfall"]
    rows = []
    for key, label in [("recurrent", "Recurrent"), ("simpler", "Simpler")]:
        m = metrics[key]
        rows.append([
            label,
            f'{m["oos_mean_loss"]:.4f}',
            f'{m["indifference_price"]:.4f}',
            f'{m["oos_realized_cvar_shortfall"]:.4f}',
        ])
    table = ax_table.table(cellText=rows, colLabels=columns, loc="center",
                           cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.65)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/e3/0"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--calibration-paths", type=int, default=1_000_000)
    parser.add_argument("--calibration-batch-size", type=int, default=25_000)
    parser.add_argument("--calibration-seed", type=int, default=271828)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bins", type=int, default=100)
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.out or (run_dir / "independent_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("e3.independent_evaluation")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = run_dir / "config.yaml"
    pnl_path = run_dir / "pnl.npz"
    r_ckpt = run_dir / "policy_recurrent.pt"
    s_ckpt = run_dir / "policy_simpler.pt"
    for path in [config_path, pnl_path, r_ckpt, s_ckpt]:
        if not path.is_file():
            raise FileNotFoundError(f"Required E3 artifact not found: {path}")

    cfg = load_yaml(config_path)
    alpha = float(cfg["alpha"])
    device = resolve_device(args.device)
    market = market_from_cfg(cfg)
    world = HestonWorld(market, device=device)
    recurrent = load_policy(r_ckpt, cfg, True, world, device)
    simpler = load_policy(s_ckpt, cfg, False, world, device)

    # The original E3 OOS losses were generated on common paths with eval_seed.
    # Reuse those frozen pathwise results; calibration uses a distinct RNG seed.
    with np.load(pnl_path) as data:
        required = {"terminal_loss", "terminal_loss_simpler"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{pnl_path} is missing arrays: {sorted(missing)}")
        oos_r = np.asarray(data["terminal_loss"], dtype=np.float64)
        oos_s = np.asarray(data["terminal_loss_simpler"], dtype=np.float64)
    if oos_r.shape != oos_s.shape or oos_r.ndim != 1:
        raise ValueError("Recurrent and simpler OOS loss arrays must be paired 1-D arrays")
    if not np.isfinite(oos_r).all() or not np.isfinite(oos_s).all():
        raise ValueError("OOS losses contain non-finite values")

    started = time.perf_counter()
    cal_r, cal_s = calibration_losses(
        world, recurrent, simpler, args.calibration_paths,
        args.calibration_batch_size, args.calibration_seed, log,
    )
    if not np.isfinite(cal_r).all() or not np.isfinite(cal_s).all():
        raise ValueError("Calibration losses contain non-finite values")

    metrics = {
        "recurrent": summarize("recurrent", cal_r, oos_r, alpha),
        "simpler": summarize("simpler", cal_s, oos_s, alpha),
    }
    hedging_error_r = metrics["recurrent"]["indifference_price"] - oos_r
    hedging_error_s = metrics["simpler"]["indifference_price"] - oos_s

    make_figure(
        hedging_error_r, hedging_error_s, metrics, alpha,
        out_dir / "figure5_e3.png", args.bins,
    )
    np.savez_compressed(
        out_dir / "evaluation_arrays.npz",
        calibration_loss_recurrent=cal_r,
        calibration_loss_simpler=cal_s,
        oos_loss_recurrent=oos_r,
        oos_loss_simpler=oos_s,
        hedging_error_recurrent=hedging_error_r,
        hedging_error_simpler=hedging_error_s,
    )
    report = {
        "schema_version": "1.0",
        "experiment": "e3",
        "source_run_dir": str(run_dir),
        "calibration_paths": int(args.calibration_paths),
        "calibration_seed": int(args.calibration_seed),
        "oos_paths": int(len(oos_r)),
        "oos_source": str(pnl_path),
        "oos_seed_from_config": int(cfg.get("train", {}).get("eval_seed", -1)),
        "alpha": alpha,
        "device": str(device),
        "runtime_sec": round(time.perf_counter() - started, 3),
        "metrics": metrics,
        "definitions": {
            "indifference_price": "CVaR_alpha(terminal_loss) estimated on independent calibration paths",
            "figure5_histogram": "indifference_price - OOS terminal_loss",
            "realized_cvar_shortfall": "CVaR_alpha(OOS terminal_loss - calibration indifference_price); lower is better",
        },
    }
    (out_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log.info("E3 independent evaluation complete")
    for name, values in metrics.items():
        log.info(
            "%s price=%.6f mean_oos_loss=%.6f realized_cvar_shortfall=%.6f",
            name, values["indifference_price"], values["oos_mean_loss"],
            values["oos_realized_cvar_shortfall"],
        )
    log.info("Figure: %s", out_dir / "figure5_e3.png")
    log.info("Report: %s", out_dir / "evaluation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
