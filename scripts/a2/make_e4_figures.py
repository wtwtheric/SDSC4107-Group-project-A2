"""Render E4 Figures 10/11 and price-asymptotic sensitivity tables from saved summaries.

No market simulation, checkpoint loading, training, or metrics-file mutation is performed.

Examples:
    python -m scripts.a2.make_e4_figures --data-root backups --seeds 0
    python -m scripts.a2.make_e4_figures --data-root backups --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

THEORETICAL_SLOPE = 2.0 / 3.0
PAPER_SLOPES = {"e4bs": 0.67, "e4h": 0.71}
EXPERIMENTS = {"e4bs": ("10", "Black–Scholes"), "e4h": ("11", "Heston")}
SUBSETS = {
    "all_5": slice(None),
    "smallest_3": slice(0, 3),
    "largest_3": slice(-3, None),
    "smallest_4": slice(0, 4),
    "largest_4": slice(-4, None),
}


def load_grid(path: Path, *, exp: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one completed five-epsilon grid; order by increasing epsilon."""
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 5:
        raise ValueError(f"{path}: expected five epsilon entries")
    if any(row.get("exp") != exp or int(row.get("seed", -1)) != seed for row in records):
        raise ValueError(f"{path}: inconsistent experiment/seed")
    if any(row.get("passed") is not True for row in records):
        raise ValueError(f"{path}: one or more epsilon runs did not pass checks")
    records.sort(key=lambda row: float(row["metrics"]["eps"]))
    eps = np.array([float(row["metrics"]["eps"]) for row in records])
    price = np.array([float(row["metrics"]["price_eps"]) for row in records])
    p0 = np.array([float(row["metrics"]["price0"]) for row in records])
    increment = np.array([float(row["metrics"]["price_increment"]) for row in records])
    if (
        not all(np.isfinite(arr).all() for arr in (eps, price, p0, increment))
        or np.any(eps <= 0)
        or np.any(increment <= 0)
        or np.any(np.diff(eps) <= 0)
    ):
        raise ValueError(f"{path}: invalid or duplicate epsilon / non-positive increment")
    if not np.allclose(price - p0, increment, rtol=1e-8, atol=1e-9):
        raise ValueError(f"{path}: increments do not agree with price_eps - price0")
    if not np.allclose(p0, p0[0], rtol=1e-10, atol=1e-10):
        raise ValueError(f"{path}: inconsistent frictionless baseline within a grid")
    return eps, increment, price


def regression(eps: np.ndarray, increment: np.ndarray) -> dict[str, float]:
    """Free-slope OLS, and a separate slope-fixed-at-2/3 reference regression."""
    if len(eps) < 3 or len(eps) != len(increment):
        raise ValueError("At least three paired observations are required")
    x, y = np.log(eps), np.log(increment)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= 0:
        raise ValueError("Cannot fit a constant log increment series")
    fixed_intercept = float(np.mean(y - THEORETICAL_SLOPE * x))
    fixed_fitted = fixed_intercept + THEORETICAL_SLOPE * x
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(1.0 - np.sum((y - fitted) ** 2) / total),
        "fixed_intercept": fixed_intercept,
        "fixed_r2": float(1.0 - np.sum((y - fixed_fitted) ** 2) / total),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    exp: str,
    eps: np.ndarray,
    increments: np.ndarray,
    per_seed_fits: list[dict[str, float]],
    output: Path,
    seeds: list[int],
) -> None:
    """Use same figure style for seed-0 and optional cross-seed aggregate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    number, title = EXPERIMENTS[exp]
    x = np.log(eps)
    y = np.log(increments)
    mean_y = np.mean(y, axis=0)
    # Report a free-slope fit to the mean log increments for visualization only;
    # statistical robustness is computed from one slope per seed, not pooled points.
    fitted = regression(eps, np.exp(mean_y))
    grid_x = np.linspace(x.min(), x.max(), 200)
    fig, ax = plt.subplots(figsize=(8.6, 5.7))
    if len(seeds) > 1:
        sd_y = np.std(y, axis=0, ddof=1)
        ax.errorbar(
            x,
            mean_y,
            yerr=sd_y,
            fmt="o",
            color="#4472c4",
            capsize=3,
            label=f"Mean log increment ± seed SD (n={len(seeds)})",
        )
    else:
        ax.scatter(x, mean_y, color="#4472c4", s=58, label="Seed 0 observations", zorder=3)
    ax.plot(
        grid_x,
        fitted["intercept"] + fitted["slope"] * grid_x,
        color="#ed7c31",
        linewidth=2.0,
        label=f"Free OLS slope = {fitted['slope']:.4f} (R² = {fitted['r2']:.4f})",
    )
    ax.plot(
        grid_x,
        fitted["fixed_intercept"] + THEORETICAL_SLOPE * grid_x,
        color="#555555",
        linestyle="--",
        linewidth=1.8,
        label="Theoretical 2/3 reference (fitted intercept)",
    )
    ax.set_title(
        f"Figure {number} replication — E4 {title}"
        + (" (multi-seed)" if len(seeds) > 1 else " (seed 0)")
    )
    ax.set_xlabel(r"$\log(\varepsilon)$")
    ax.set_ylabel(r"$\log(p_{\varepsilon}-p_0)$")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=0.2)
    ax.text(
        0.99,
        0.02,
        f"Paper slope ≈ {PAPER_SLOPES[exp]:.2f}  |  Theoretical 2/3",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figures(
    *, data_root: Path, figures_dir: Path, report_dir: Path, seeds: list[int]
) -> list[Path]:
    """Generate plots and auditable per-seed sensitivity tables from existing JSON."""
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Supply unique, non-negative seeds")
    output_paths: list[Path] = []
    sensitivity: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    robustness: list[dict[str, Any]] = []

    for exp, (figure_no, _title) in EXPERIMENTS.items():
        eps_ref: np.ndarray | None = None
        all_increments: list[np.ndarray] = []
        slopes: list[float] = []
        for seed in seeds:
            source = data_root / exp / f"seed_{seed}_epsilon_summary.json"
            if not source.is_file():
                raise FileNotFoundError(f"Missing E4 summary: {source}")
            eps, increment, price = load_grid(source, exp=exp, seed=seed)
            if eps_ref is None:
                eps_ref = eps
            elif not np.allclose(eps, eps_ref, rtol=0, atol=1e-12):
                raise ValueError(f"{source}: epsilon grid differs between seeds")
            all_increments.append(increment)
            for e, inc, p in zip(eps, increment, price, strict=True):
                point_rows.append(
                    {
                        "experiment": exp,
                        "seed": seed,
                        "epsilon": float(e),
                        "price_eps": float(p),
                        "price_increment": float(inc),
                        "source_summary": str(source),
                    }
                )
            for subset, selection in SUBSETS.items():
                fit = regression(eps[selection], increment[selection])
                sensitivity.append(
                    {
                        "experiment": exp,
                        "seed": seed,
                        "subset": subset,
                        "n_points": len(eps[selection]),
                        "min_epsilon": float(eps[selection].min()),
                        "max_epsilon": float(eps[selection].max()),
                        "slope": fit["slope"],
                        "r2": fit["r2"],
                        "intercept": fit["intercept"],
                        "theoretical_slope": THEORETICAL_SLOPE,
                        "reference_fixed_r2": fit["fixed_r2"],
                        "paper_reported_slope": PAPER_SLOPES[exp],
                        "source_summary": str(source),
                    }
                )
                if subset == "all_5":
                    slopes.append(fit["slope"])
        assert eps_ref is not None
        arr = np.stack(all_increments, axis=0)
        output = figures_dir / f"fig{figure_no}_e4_{'bs' if exp == 'e4bs' else 'heston'}.png"
        _plot(exp, eps_ref, arr, [], output, seeds)
        output_paths.append(output)
        robustness.append(
            {
                "experiment": exp,
                "n_seeds": len(seeds),
                "seeds": ";".join(str(s) for s in seeds),
                "slope_mean": float(np.mean(slopes)),
                "slope_sd": float(np.std(slopes, ddof=1)) if len(slopes) > 1 else float("nan"),
                "theoretical_slope": THEORETICAL_SLOPE,
                "paper_reported_slope": PAPER_SLOPES[exp],
            }
        )

    sensitivity_csv = report_dir / "e4_epsilon_sensitivity.csv"
    prices_csv = report_dir / "e4_price_increments.csv"
    robustness_csv = report_dir / "e4_seed_slopes.csv"
    _write_csv(sensitivity_csv, sensitivity)
    _write_csv(prices_csv, point_rows)
    _write_csv(robustness_csv, robustness)
    output_paths.extend([sensitivity_csv, prices_csv, robustness_csv])

    lines = [
        "# E4 epsilon-subset slope sensitivity",
        "",
        "Computed from saved epsilon summaries, never from retraining or simulated new samples.",
        "Slopes are per seed; n=3 subsets are descriptive, not robust asymptotic estimators.",
        "",
        "| Market | Seed | Subset | Slope | R² |",
        "|---|---:|---|---:|---:|",
    ]
    for row in sensitivity:
        lines.append(
            f"| {row['experiment']} | {row['seed']} | {row['subset']} "
            f"| {row['slope']:.4f} | {row['r2']:.4f} |"
        )
    lines += [
        "",
        "Theoretical exponent: 2/3 ≈ 0.6667. The fixed-slope reference",
        "intercept is fitted separately; it is not an independently estimated exponent.",
        "",
    ]
    md_path = report_dir / "e4_epsilon_sensitivity.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    output_paths.append(md_path)
    return output_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("backups"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/e4"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    for path in make_figures(
        data_root=args.data_root,
        figures_dir=args.figures_dir,
        report_dir=args.report_dir,
        seeds=args.seeds,
    ):
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
