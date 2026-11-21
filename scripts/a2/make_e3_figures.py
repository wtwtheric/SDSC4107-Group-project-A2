"""Render the E3 Figure 5 from saved independent calibration and paired OOS losses.

This module owns only plotting and reads saved files; it never simulates paths,
updates metrics, or loads model checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def make_figure(
    recurrent_error: np.ndarray,
    simpler_error: np.ndarray,
    metrics: dict[str, dict[str, float]],
    alpha: float,
    output: Path,
    bins: int,
) -> None:
    """Save the E3 Figure 5 histogram and comparison table at 300 dpi."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
        alpha=0.55,
        label="recurrent",
        color="#4472C4",
        edgecolor="none",
    )
    ax.hist(
        simpler_error,
        bins=edges,
        alpha=0.55,
        label="simpler",
        color="#ed7c31",
        edgecolor="none",
    )
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xlabel(r"Terminal hedging error  $p_0 - L_{\mathrm{raw}}$")
    ax.set_ylabel("Number of OOS paths")
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.2)

    table_ax = fig.add_subplot(grid[1])
    table_ax.axis("off")
    columns = ["Model", "Mean Loss*", "Price", "Realized CVaR"]
    rows = [
        [
            "Recurrent",
            f"{metrics['recurrent']['oos_mean_excess_loss_diagnostic']:.6f}",
            f"{metrics['recurrent']['indifference_price']:.4f}",
            f"{metrics['recurrent']['oos_realized_cvar_shortfall']:.4f}",
        ],
        [
            "Simpler",
            f"{metrics['simpler']['oos_mean_excess_loss_diagnostic']:.6f}",
            f"{metrics['simpler']['indifference_price']:.4f}",
            f"{metrics['simpler']['oos_realized_cvar_shortfall']:.4f}",
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
        0.5,
        0.04,
        "* Mean Loss proxy: E[max(L_raw - p0, 0)]; not confirmed as the paper's Mean Loss.",
        ha="center",
        va="bottom",
        fontsize=7,
        transform=table_ax.transAxes,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_saved(*, run_dir: Path, output: Path | None = None, bins: int = 100) -> Path:
    """Read only evaluation.json + original paired losses in pnl.npz."""
    report_path = run_dir / "evaluation.json"
    pnl_path = run_dir / "pnl.npz"
    if not report_path.is_file() or not pnl_path.is_file():
        raise FileNotFoundError(
            f"Figure 5 requires evaluation.json and pnl.npz in {run_dir}. "
            "The GitHub backups currently omit pnl.npz; use the Kaggle run directory."
        )
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    with np.load(pnl_path, allow_pickle=False) as saved:
        recurrent = np.asarray(saved["terminal_loss"], dtype=np.float64)
        simpler = np.asarray(saved["terminal_loss_simpler"], dtype=np.float64)
    if recurrent.ndim != 1 or simpler.shape != recurrent.shape:
        raise ValueError("Paired OOS losses must be 1D arrays of the same length")
    if not np.isfinite(recurrent).all() or not np.isfinite(simpler).all():
        raise ValueError("OOS losses contain non-finite values")
    metrics = report["metrics"]
    out = output or run_dir / "fig_05_e3.png"
    make_figure(
        metrics["recurrent"]["indifference_price"] - recurrent,
        metrics["simpler"]["indifference_price"] - simpler,
        metrics,
        float(report["alpha"]),
        out,
        bins,
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/e3/0"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bins", type=int, default=100)
    args = parser.parse_args()
    print(f"saved {plot_saved(run_dir=args.run_dir, output=args.output, bins=args.bins)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
