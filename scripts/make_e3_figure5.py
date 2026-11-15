"""Create the E3 paper-style Figure 5 from independent evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.common.config import REPO_ROOT

import numpy as np


def make_figure5(*, evaluation_dir: Path, figures_dir: Path) -> Path:
    """Plot common OOS hedging errors and the independently calibrated E3 summary table."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_path = evaluation_dir / "metrics.json"
    pnl_path = evaluation_dir / "pnl.npz"
    if not metrics_path.is_file() or not pnl_path.is_file():
        raise FileNotFoundError(
            f"Expected metrics.json and pnl.npz in {evaluation_dir}; run scripts.evaluate_e3 first."
        )
    payload: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    with np.load(pnl_path, allow_pickle=False) as archive:
        required = ("hedging_error_recurrent", "hedging_error_simpler")
        missing = [key for key in required if key not in archive]
        if missing:
            raise KeyError(f"E3 evaluation pnl.npz is missing arrays: {missing}")
        recurrent = np.asarray(archive[required[0]], dtype=np.float64)
        simpler = np.asarray(archive[required[1]], dtype=np.float64)

    if recurrent.ndim != 1 or simpler.ndim != 1 or recurrent.shape != simpler.shape:
        raise ValueError("Figure 5 requires paired one-dimensional OOS hedging-error arrays")
    if not np.isfinite(recurrent).all() or not np.isfinite(simpler).all():
        raise ValueError("Figure 5 hedging-error arrays contain non-finite values")

    combined = np.concatenate([recurrent, simpler])
    lo, hi = np.quantile(combined, [0.001, 0.999])
    if not np.isfinite(lo + hi) or hi <= lo:
        lo, hi = float(combined.min()) - 1e-6, float(combined.max()) + 1e-6
    bins = np.linspace(lo, hi, 151)

    fig = plt.figure(figsize=(8.8, 6.8))
    grid = fig.add_gridspec(2, 1, height_ratios=(4.0, 1.35), hspace=0.27)
    ax = fig.add_subplot(grid[0])
    ax.hist(recurrent, bins=bins, alpha=0.55, label="recurrent", color="#4c78a8")
    ax.hist(simpler, bins=bins, alpha=0.45, label="simpler", color="#f58518")
    ax.set_xlabel(r"Terminal hedging error  $H=p_{cal}-L_{OOS}$")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    ax.text(
        0.99, 0.97,
        f"alpha={payload['calibration']['alpha']:.2f} | "
        f"calibration n={payload['calibration']['n_paths']:,} | "
        f"OOS n={payload['oos']['n_paths']:,}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.3",
    )

    ax_table = fig.add_subplot(grid[1])
    ax_table.axis("off")
    rows = []
    for label in ("recurrent", "simpler"):
        row = metrics[label]
        rows.append([
            label,
            f"{row['mean_abs_hedging_error']:.4f}",
            f"{row['price']:.4f}",
            f"{row['realized_cvar']:.4f}",
        ])
    table = ax_table.table(
        cellText=rows,
        colLabels=("Strategy", "Mean |H|*", "Price", "Realized CVaR"),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)

    fig.text(
        0.01, 0.012,
        "*Mean |H| is reported as a transparent proxy because the paper's Figure 5 caption "
        "does not define its 'Mean Loss' column. Realized CVaR is the lower-tail CVaR of H. "
        "Histogram displays the 0.1st–99.9th percentile range; all paths enter the metrics.",
        fontsize=7.2, color="0.25", va="bottom",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))

    figures_dir.mkdir(parents=True, exist_ok=True)
    output = figures_dir / "fig05_e3_recurrent_vs_simpler.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render E3 Figure 5.")
    parser.add_argument(
        "--evaluation-dir", type=Path,
        default=Path("results/e3/0/independent_evaluation"),
    )
    parser.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "figures")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    make_figure5(evaluation_dir=args.evaluation_dir, figures_dir=args.figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
