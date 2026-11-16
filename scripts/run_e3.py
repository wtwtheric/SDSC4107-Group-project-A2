"""Run E3 end-to-end: train (or load checkpoints), calibrate, OOS-evaluate, and plot Figure 5."""

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
from src.common.config import dump_runtime_config, load_config
from src.common.logging import human_secs, setup_logger
from src.common.metrics import cvar, save_pnl, write_metrics
from src.common.seed import resolve_device, seed_everything
from src.replication.e1 import market_from_cfg, policy_from_cfg
from src.replication.e3 import run as train_and_evaluate
from src.replication.env_heston import HestonWorld


def empirical_cvar(values: np.ndarray, alpha: float) -> float:
    """Return empirical upper-tail CVaR for a one-dimensional loss sample."""
    return float(cvar(torch.as_tensor(values, dtype=torch.float64), alpha))


def summarize(
    calibration_loss: np.ndarray, oos_loss: np.ndarray, alpha: float
) -> dict[str, float]:
    """Calibrate price on one sample and assess realized risk on an independent sample."""
    price = empirical_cvar(calibration_loss, alpha)
    oos_cvar = empirical_cvar(oos_loss, alpha)
    hedging_error = price - oos_loss
    centered_loss = oos_loss - price
    return {
        "calibration_mean_raw_loss": float(calibration_loss.mean()),
        "indifference_price": price,
        "oos_mean_raw_loss": float(oos_loss.mean()),
        "oos_signed_terminal_hedging_loss": float(centered_loss.mean()),
        "oos_mean_hedging_error": float(hedging_error.mean()),
        "oos_mean_excess_loss_diagnostic": float(np.maximum(centered_loss, 0.0).mean()),
        "oos_mean_positive_hedging_error": float(np.maximum(hedging_error, 0.0).mean()),
        "oos_realized_cvar_loss": oos_cvar,
        "oos_realized_cvar_shortfall": empirical_cvar(oos_loss - price, alpha),
        "oos_hedging_error_q01": float(np.quantile(hedging_error, 0.01)),
        "oos_hedging_error_median": float(np.median(hedging_error)),
        "oos_hedging_error_q99": float(np.quantile(hedging_error, 0.99)),
    }


def load_policy(
    checkpoint: Path,
    cfg: dict[str, Any],
    recurrent: bool,
    world: HestonWorld,
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate a frozen E3 policy and load its saved state dictionary."""
    policy = policy_from_cfg(
        cfg, world.market.n_features, world.market.d, device, recurrent=recurrent
    )
    state = torch.load(checkpoint, map_location=device, weights_only=True)
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
    """Evaluate both frozen policies on common, independently seeded calibration paths."""
    seed_everything(seed)
    recurrent_parts: list[np.ndarray] = []
    simpler_parts: list[np.ndarray] = []
    done = 0
    while done < n_paths:
        size = min(batch_size, n_paths - done)
        paths = world.simulate(size)
        features = world.features(paths)
        recurrent_positions = recurrent.forward_path(features)
        simpler_positions = simpler.forward_path(features)
        recurrent_parts.append(
            world.terminal_loss(recurrent_positions, paths).cpu().numpy()
        )
        simpler_parts.append(
            world.terminal_loss(simpler_positions, paths).cpu().numpy()
        )
        done += size
        if done % max(batch_size * 4, batch_size) == 0 or done == n_paths:
            log.info("calibration paths %d/%d", done, n_paths)
    return np.concatenate(recurrent_parts), np.concatenate(simpler_parts)


def make_figure(
    recurrent_error: np.ndarray,
    simpler_error: np.ndarray,
    metrics: dict[str, dict[str, float]],
    alpha: float,
    output: Path,
    bins: int,
) -> None:
    """Save the E3 Figure 5 histogram and comparison table at 300 dpi."""
    pooled = np.concatenate([recurrent_error, simpler_error])
    lo, hi = np.quantile(pooled, [0.001, 0.999])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("Cannot construct histogram range from hedging errors")
    edges = np.linspace(lo, hi, bins + 1)

    fig = plt.figure(figsize=(11.5, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[4.5, 1.45])
    ax = fig.add_subplot(grid[0])
    ax.hist(
        recurrent_error,
        bins=edges,
        alpha=0.62,
        label="recurrent",
        color="#4472C4",
        edgecolor="none",
    )
    ax.hist(
        simpler_error,
        bins=edges,
        alpha=0.52,
        label="simpler",
        color="#808080",
        edgecolor="none",
    )
    ax.axvline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.set_xlabel(r"Terminal hedging error  $p_0 - L_T$")
    ax.set_ylabel("Number of OOS paths")
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.2)

    table_ax = fig.add_subplot(grid[1])
    table_ax.axis("off")
    columns = ["Model", "Mean Loss", "Price", "Realized CVaR"]
    rows = [
        [
            "Recurrent",
            f'{metrics["recurrent"]["oos_signed_terminal_hedging_loss"]:.4f}',
            f'{metrics["recurrent"]["indifference_price"]:.4f}',
            f'{metrics["recurrent"]["oos_realized_cvar_shortfall"]:.4f}',
        ],
        [
            "Simpler",
            f'{metrics["simpler"]["oos_signed_terminal_hedging_loss"]:.4f}',
            f'{metrics["simpler"]["indifference_price"]:.4f}',
            f'{metrics["simpler"]["oos_realized_cvar_shortfall"]:.4f}',
        ],
    ]
    table = table_ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    table_ax.text(
        0.5, 0.04,
        "Mean Loss = E[L_T]; see documentation for the operational interpretation.",
        ha="center", va="bottom", fontsize=7, transform=table_ax.transAxes,
    )
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def update_saved_pnl(
    pnl_path: Path,
    calibration_recurrent: np.ndarray,
    calibration_simpler: np.ndarray,
    hedging_error_recurrent: np.ndarray,
    hedging_error_simpler: np.ndarray,
) -> None:
    """Append calibration and centered OOS arrays without dropping existing PnL fields."""
    with np.load(pnl_path) as saved:
        arrays = {key: np.asarray(saved[key]) for key in saved.files}
    terminal_loss = arrays.pop("terminal_loss")
    arrays.update(
        {
            "calibration_loss_recurrent": calibration_recurrent,
            "calibration_loss_simpler": calibration_simpler,
            "hedging_error_recurrent": hedging_error_recurrent,
            "hedging_error_simpler": hedging_error_simpler,
        }
    )
    save_pnl(pnl_path, terminal_loss, **arrays)


def run_independent_evaluation(
    run_dir: Path,
    cfg: dict[str, Any],
    device: torch.device,
    log: logging.Logger,
    calibration_paths_override: int | None = None,
    calibration_batch_override: int | None = None,
    calibration_seed_override: int | None = None,
    bins: int = 100,
) -> dict[str, Any]:
    """Load existing E3 artifacts, independently calibrate prices, and evaluate OOS risk."""
    config = cfg
    train_cfg = config.get("train", {}) or {}
    alpha = float(config.get("alpha", 0.99))
    n_calibration = int(
        calibration_paths_override
        or train_cfg.get("calibration_paths", train_cfg.get("eval_paths", 1_000_000))
    )
    batch_size = int(
        calibration_batch_override
        or train_cfg.get("calibration_batch_size", train_cfg.get("eval_batch_size", 25_000))
    )
    calibration_seed = int(
        calibration_seed_override
        if calibration_seed_override is not None
        else train_cfg.get("calibration_seed", 271828)
    )
    oos_seed = int(train_cfg.get("eval_seed", 314159))
    if calibration_seed == oos_seed:
        raise ValueError("Calibration and OOS seeds must differ to ensure independent samples")
    if n_calibration < 1 or batch_size < 1:
        raise ValueError("Calibration path count and batch size must be positive")

    pnl_path = run_dir / "pnl.npz"
    recurrent_path = run_dir / "policy_recurrent.pt"
    simpler_path = run_dir / "policy_simpler.pt"
    for path in (pnl_path, recurrent_path, simpler_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required E3 artifact not found: {path}")

    with np.load(pnl_path) as saved:
        required = {"terminal_loss", "terminal_loss_simpler"}
        missing = required.difference(saved.files)
        if missing:
            raise KeyError(f"{pnl_path} is missing arrays: {sorted(missing)}")
        oos_recurrent = np.asarray(saved["terminal_loss"], dtype=np.float64)
        oos_simpler = np.asarray(saved["terminal_loss_simpler"], dtype=np.float64)
    if oos_recurrent.ndim != 1 or oos_recurrent.shape != oos_simpler.shape:
        raise ValueError("Recurrent and simpler OOS losses must be paired 1-D arrays")
    if not np.isfinite(oos_recurrent).all() or not np.isfinite(oos_simpler).all():
        raise ValueError("OOS losses contain non-finite values")

    market = market_from_cfg(config)
    world = HestonWorld(market, device=device)
    recurrent = load_policy(recurrent_path, config, True, world, device)
    simpler = load_policy(simpler_path, config, False, world, device)

    started = time.perf_counter()
    calibration_recurrent, calibration_simpler = calibration_losses(
        world,
        recurrent,
        simpler,
        n_calibration,
        batch_size,
        calibration_seed,
        log,
    )
    if not np.isfinite(calibration_recurrent).all() or not np.isfinite(
        calibration_simpler
    ).all():
        raise ValueError("Calibration losses contain non-finite values")

    metrics = {
        "recurrent": summarize(calibration_recurrent, oos_recurrent, alpha),
        "simpler": summarize(calibration_simpler, oos_simpler, alpha),
    }
    error_recurrent = metrics["recurrent"]["indifference_price"] - oos_recurrent
    error_simpler = metrics["simpler"]["indifference_price"] - oos_simpler

    make_figure(
        error_recurrent,
        error_simpler,
        metrics,
        alpha,
        run_dir / "fig_05_e3.png",
        bins,
    )
    update_saved_pnl(
        pnl_path,
        calibration_recurrent,
        calibration_simpler,
        error_recurrent,
        error_simpler,
    )

    report = {
        "schema_version": "1.0",
        "experiment": "e3",
        "source_run_dir": str(run_dir),
        "calibration_paths": n_calibration,
        "calibration_seed": calibration_seed,
        "oos_paths": int(len(oos_recurrent)),
        "oos_source": str(pnl_path),
        "oos_seed": oos_seed,
        "alpha": alpha,
        "device": str(device),
        "runtime_sec": round(time.perf_counter() - started, 3),
        "metrics": metrics,
        "definitions": {
            "indifference_price": (
                "CVaR_alpha(terminal_loss) estimated on independent calibration paths"
            ),
            "figure5_histogram": "indifference_price - OOS terminal_loss",
            "realized_cvar_shortfall": (
                "CVaR_alpha(OOS terminal_loss - calibration indifference_price)"
            ),
        },
    }
    (run_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("E3 independent evaluation complete")
    for name, values in metrics.items():
        log.info(
            "%s price=%.6f mean_raw_loss=%.6f mean_excess_candidate=%.6f realized_cvar_shortfall=%.6f",
            name,
            values["indifference_price"],
            values["oos_mean_raw_loss"],
            values["oos_signed_terminal_hedging_loss"],
            values["oos_realized_cvar_shortfall"],
        )
    log.info("Figure: %s", run_dir / "fig_05_e3.png")
    log.info("Report: %s", run_dir / "evaluation.json")
    return {
        "report": report,
        "calibration_seed": calibration_seed,
        "oos_seed": oos_seed,
        "oos_paths": int(len(oos_recurrent)),
    }


def apply_evaluation_to_metrics(
    metrics: dict[str, float],
    checks: dict[str, bool],
    report: dict[str, Any],
) -> tuple[dict[str, float], dict[str, bool]]:
    """Map independent calibration/OOS results to the repository's E3 metrics schema."""
    recurrent = report["metrics"]["recurrent"]
    simpler = report["metrics"]["simpler"]
    metrics.update(
        {
            "price_recurrent": recurrent["indifference_price"],
            "price_simpler": simpler["indifference_price"],
            "cvar_recurrent": recurrent["oos_realized_cvar_loss"],
            "cvar_simpler": simpler["oos_realized_cvar_loss"],
            "mean_loss_recurrent": recurrent["oos_signed_terminal_hedging_loss"],
            "mean_loss_simpler": simpler["oos_signed_terminal_hedging_loss"],
            "figure5_mean_loss_recurrent": recurrent["oos_signed_terminal_hedging_loss"],
            "figure5_mean_loss_simpler": simpler["oos_signed_terminal_hedging_loss"],
            "signed_terminal_hedging_loss_recurrent": recurrent["oos_signed_terminal_hedging_loss"],
            "signed_terminal_hedging_loss_simpler": simpler["oos_signed_terminal_hedging_loss"],
            "cvar_shortfall_recurrent": recurrent["oos_realized_cvar_shortfall"],
            "cvar_shortfall_simpler": simpler["oos_realized_cvar_shortfall"],
        }
    )
    checks.update(
        {
            "independent_calibration_oos_seeds": (
                report["calibration_seed"] != report["oos_seed"]
            ),
            "recurrent_price_in_paper_band": (
                5.2 <= recurrent["indifference_price"] <= 5.8
            ),
            "simpler_price_in_paper_band": (
                6.4 <= simpler["indifference_price"] <= 7.1
            ),
            "recurrent_price_lower_than_simpler": (
                recurrent["indifference_price"] < simpler["indifference_price"]
            ),
            "recurrent_oos_cvar_lower_than_simpler": (
                recurrent["oos_realized_cvar_loss"] < simpler["oos_realized_cvar_loss"]
            ),
        }
    )
    return metrics, checks


def update_metrics_file(
    run_dir: Path,
    metrics: dict[str, float],
    checks: dict[str, bool],
    evaluation: dict[str, Any],
) -> None:
    """Update metrics after eval-only while preserving the original training provenance."""
    path = run_dir / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Cannot update missing metrics file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("metrics", {}).update(metrics)
    payload.setdefault("checks", {}).update(checks)
    payload["passed"] = bool(payload["checks"]) and all(payload["checks"].values())
    payload["independent_evaluation"] = {
        "calibration_paths": evaluation["calibration_paths"],
        "calibration_seed": evaluation["calibration_seed"],
        "oos_paths": evaluation["oos_paths"],
        "oos_seed": evaluation["oos_seed"],
        "device": evaluation["device"],
        "runtime_sec": evaluation["runtime_sec"],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/exp/e3.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate existing checkpoints and OOS pnl.npz.",
    )
    parser.add_argument("--calibration-paths", type=int, default=None)
    parser.add_argument("--calibration-batch-size", type=int, default=None)
    parser.add_argument("--calibration-seed", type=int, default=None)
    parser.add_argument("--bins", type=int, default=100)
    args = parser.parse_args()

    folder = args.run_dir or (Path(args.out) / "e3" / str(args.seed))
    if args.eval_only:
        config_path = folder / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Saved E3 config not found: {config_path}")
        cfg = load_config(config_path)
    else:
        cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    device = resolve_device(cfg["device"])

    if args.eval_only:
        log = setup_logger("a2.e3.eval", folder / "evaluation.log")
        started = time.perf_counter()
        log.info("E3 eval-only run_dir=%s device=%s", folder, device)
        evaluation = run_independent_evaluation(
            folder,
            cfg,
            device,
            log,
            args.calibration_paths,
            args.calibration_batch_size,
            args.calibration_seed,
            args.bins,
        )
        payload = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        metrics, checks = apply_evaluation_to_metrics(
            payload.get("metrics", {}),
            payload.get("checks", {}),
            evaluation["report"],
        )
        update_metrics_file(folder, metrics, checks, evaluation["report"])
        log.info("E3 eval-only completed runtime=%s", human_secs(time.perf_counter() - started))
        return 0

    if args.resume and (folder / "metrics.json").exists():
        print(f"E3 seed {args.seed} already complete; skipping.")
        return 0
    if (folder / "metrics.json").exists():
        raise FileExistsError(f"{folder} already contains metrics; use --resume or a fresh --out")
    folder.mkdir(parents=True, exist_ok=True)
    log = setup_logger(f"a2.e3.{args.seed}", folder / "run.log")
    started = time.perf_counter()
    log.info("E3 seed=%d device=%s config=%s", args.seed, device, args.config)

    result = train_and_evaluate(cfg, args.seed, device, log)
    dump_runtime_config(cfg, folder / "config.yaml")
    save_pnl(folder / "pnl.npz", result["terminal_loss"], **result.get("extra_pnl", {}))
    for name, state in result.get("artifacts", {}).items():
        torch.save(state, folder / name)

    evaluation = run_independent_evaluation(
        folder,
        cfg,
        device,
        log,
        args.calibration_paths,
        args.calibration_batch_size,
        args.calibration_seed,
        args.bins,
    )
    result["metrics"], result["checks"] = apply_evaluation_to_metrics(
        result["metrics"], result["checks"], evaluation["report"]
    )
    record = write_metrics(
        folder / "metrics.json",
        exp="e3",
        seed=args.seed,
        cfg=cfg,
        metrics=result["metrics"],
        checks=result["checks"],
        runtime_sec=time.perf_counter() - started,
        device=device,
        n_paths=result["n_paths"],
    )
    log.info(
        "E3 completed runtime=%s passed=%s",
        human_secs(time.perf_counter() - started),
        record["passed"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
