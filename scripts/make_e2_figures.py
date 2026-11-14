"""Create paper-style E2 Figures 6 and 7 in the repository-root figures/ directory."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from src.common.config import REPO_ROOT


def _cvar_np(values: np.ndarray, alpha: float) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not 0.0 < alpha < 1.0:
        raise ValueError("CVaR requires a non-empty array and alpha in (0, 1)")
    count = max(1, math.ceil((1.0 - alpha) * flat.size - 1e-12))
    return float(np.partition(flat, flat.size - count)[-count:].mean())


def build_summary(loss50: np.ndarray, loss99: np.ndarray, q: float) -> list[dict[str, float | str]]:
    """Paper-table diagnostics; Mean Loss is operationalized as mean |q-L|."""
    rows: list[dict[str, float | str]] = []
    for label, loss in (("0.50-CVaR", loss50), ("0.99-CVaR", loss99)):
        excess_loss = np.asarray(loss, dtype=np.float64) - q
        rows.append(
            {
                "strategy": label,
                "mean_abs_hedging_error": float(np.mean(np.abs(excess_loss))),
                "realized_cvar_50": _cvar_np(excess_loss, 0.50),
                "realized_cvar_99": _cvar_np(excess_loss, 0.99),
            }
        )
    return rows


def _load_inputs(evaluation_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    metrics_path = evaluation_dir / "metrics.json"
    pnl_path = evaluation_dir / "pnl.npz"
    if not metrics_path.is_file() or not pnl_path.is_file():
        raise FileNotFoundError(
            f"Expected metrics.json and pnl.npz in evaluation directory: {evaluation_dir}"
        )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", payload)
    with np.load(pnl_path, allow_pickle=False) as archive:
        if "terminal_loss_alpha50" not in archive or "terminal_loss_alpha99" not in archive:
            raise KeyError(
                "E2 PnL archive must contain terminal_loss_alpha50 and terminal_loss_alpha99"
            )
        loss50 = archive["terminal_loss_alpha50"].astype(np.float64, copy=False)
        loss99 = archive["terminal_loss_alpha99"].astype(np.float64, copy=False)
    if loss50.shape != loss99.shape:
        raise ValueError(f"E2 loss arrays must share shape, got {loss50.shape} and {loss99.shape}")
    return metrics, loss50, loss99


def _plot_histogram(
    plt: Any,
    values_a: np.ndarray,
    values_b: np.ndarray,
    labels: tuple[str, str],
    xlabel: str,
    title: str,
    output: Path,
    table_rows: list[dict[str, float | str]] | None = None,
) -> None:
    combined = np.concatenate([values_a, values_b])
    lo, hi = np.quantile(combined, [0.001, 0.999])
    if not np.isfinite(lo + hi) or hi <= lo:
        lo = float(np.min(combined)) - 1e-6
        hi = float(np.max(combined)) + 1e-6
    bins = np.linspace(lo, hi, 151)

    if table_rows is None:
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        ax.hist(values_a, bins=bins, histtype="step", linewidth=1.7, label=labels[0])
        ax.hist(values_b, bins=bins, histtype="step", linewidth=1.7, label=labels[1])
        ax.set_ylabel("Frequency")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        fig.text(
            0.99,
            0.01,
            "Displayed range: 0.1st–99.9th percentiles; all paths used for metrics.",
            ha="right",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.035, 1, 1))
    else:
        fig = plt.figure(figsize=(8.8, 6.8))
        grid = fig.add_gridspec(2, 1, height_ratios=(4, 1.25), hspace=0.28)
        ax = fig.add_subplot(grid[0])
        ax.hist(values_a, bins=bins, histtype="step", linewidth=1.7, label=labels[0])
        ax.hist(values_b, bins=bins, histtype="step", linewidth=1.7, label=labels[1])
        ax.set_ylabel("Frequency")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        ax_table = fig.add_subplot(grid[1])
        ax_table.axis("off")
        cell_text = [
            [
                str(row["strategy"]),
                f"{float(row['mean_abs_hedging_error']):.4f}",
                f"{float(row['realized_cvar_50']):.4f}",
                f"{float(row['realized_cvar_99']):.4f}",
            ]
            for row in table_rows
        ]
        table = ax_table.table(
            cellText=cell_text,
            colLabels=("Strategy", "Mean |error|", "Realized CVaR 50", "Realized CVaR 99"),
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.35)
        fig.text(
            0.99,
            0.01,
            "Mean |error| = mean(|q − L|); CVaR uses the upper tail of L − q.",
            ha="right",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.035, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figures(*, evaluation_dir: Path, figures_dir: Path) -> tuple[Path, Path, Path]:
    # Lazy import lets CI test the summary logic without installing matplotlib.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics, loss50, loss99 = _load_inputs(evaluation_dir)
    q = float(metrics["q"])
    price50 = float(metrics["price_alpha50"])
    price99 = float(metrics["price_alpha99"])

    # Figure 6: each strategy is charged its own indifference premium.
    fig06 = figures_dir / "fig06_e2_own_price.png"
    _plot_histogram(
        plt,
        price99 - loss99,
        price50 - loss50,
        ("0.99-CVaR policy (own price)", "0.50-CVaR policy (own price)"),
        "Hedging error at own premium (pα − Z + hedge gain)",
        "E2: Own risk-adjusted premium",
        fig06,
    )

    # Figure 7: both strategies are charged the same risk-neutral premium q.
    summary = build_summary(loss50, loss99, q)
    fig07 = figures_dir / "fig07_e2_common_q.png"
    _plot_histogram(
        plt,
        q - loss99,
        q - loss50,
        ("0.99-CVaR policy (common q)", "0.50-CVaR policy (common q)"),
        "Hedging error at common risk-neutral premium (q − Z + hedge gain)",
        "E2: Common risk-neutral premium",
        fig07,
        table_rows=summary,
    )

    summary_csv = evaluation_dir / "fig07_e2_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "strategy",
                "mean_abs_hedging_error",
                "realized_cvar_50",
                "realized_cvar_99",
            ),
        )
        writer.writeheader()
        writer.writerows(summary)

    print(f"saved {fig06}")
    print(f"saved {fig07}")
    print(f"saved {summary_csv}")
    return fig06, fig07, summary_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render canonical E2 Figures 6 and 7.")
    parser.add_argument("--evaluation-dir", type=Path, default=Path("results/e2/0"))
    parser.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "figures")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    make_figures(evaluation_dir=args.evaluation_dir, figures_dir=args.figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
