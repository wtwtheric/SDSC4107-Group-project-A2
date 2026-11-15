"""Create 3D surfaces and 2D heatmaps for the E2 call-spread strategy shapes."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d


def _policy_surface(data: dict[str, np.ndarray], key: str) -> tuple[np.ndarray, np.ndarray]:
    s = data["visual_spot"].astype(float)
    v = data["visual_variance"].astype(float)
    delta = data[key].astype(float)
    sg = data["model_spot_grid"].astype(float)
    vg = data["model_variance_grid"].astype(float)
    # Use exactly the same regular grid as the model benchmark.
    se = np.r_[sg - (sg[1] - sg[0]) / 2, sg[-1] + (sg[-1] - sg[-2]) / 2]
    ve = np.r_[vg - (vg[1] - vg[0]) / 2, vg[-1] + (vg[-1] - vg[-2]) / 2]
    med, _, _, _ = binned_statistic_2d(s, v, delta, statistic="median", bins=[se, ve])
    counts, _, _ = np.histogram2d(s, v, bins=[se, ve])
    valid = np.isfinite(med)
    if valid.sum() < 10:
        raise RuntimeError("Too few occupied (S,V) bins to build a reliable policy surface")
    ss, vv = np.meshgrid(sg, vg, indexing="ij")
    points = np.column_stack([ss[valid], vv[valid]])
    values = med[valid]
    filled = griddata(points, values, (ss, vv), method="linear")
    missing = ~np.isfinite(filled)
    if missing.any():
        filled[missing] = griddata(points, values, (ss[missing], vv[missing]), method="nearest")
    return filled, counts


def _plot_3d(
    out: Path, title: str, s: np.ndarray, v: np.ndarray, surfaces: list[tuple[str, np.ndarray]]
) -> None:
    ss, vv = np.meshgrid(s, v, indexing="ij")
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    for idx, (label, z) in enumerate(surfaces, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        ax.plot_surface(
            ss, vv, z, cmap="viridis", linewidth=0, antialiased=True, rstride=1, cstride=1
        )
        ax.set_title(label)
        ax.set_xlabel("Spot price S")
        ax.set_ylabel("Variance V")
        ax.set_zlabel("Hedge position")
        ax.view_init(elev=28, azim=-135)
    fig.suptitle(title, fontsize=15)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_heatmaps(
    out: Path, title: str, s: np.ndarray, v: np.ndarray, surfaces: list[tuple[str, np.ndarray]]
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, (label, z) in zip(axes.flat, surfaces, strict=True):
        mesh = ax.pcolormesh(s, v, z.T, shading="auto", cmap="viridis")
        ax.set_title(label)
        ax.set_xlabel("Spot price S")
        ax.set_ylabel("Variance V")
        fig.colorbar(mesh, ax=ax, label="Hedge position")
    fig.suptitle(title, fontsize=15)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/e2cs")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result_root = Path(args.results)
    figure_root = Path(args.figures)
    figure_root.mkdir(parents=True, exist_ok=True)
    data95 = dict(np.load(result_root / "0.95" / str(args.seed) / "pnl.npz"))
    data99 = dict(np.load(result_root / "0.99" / str(args.seed) / "pnl.npz"))
    sg = data95["model_spot_grid"]
    vg = data95["model_variance_grid"]
    model_stock = data95["model_stock_delta_grid"]
    model_varswap = data95["model_varswap_delta_grid"]
    stock95, counts95 = _policy_surface(data95, "visual_stock_delta")
    stock99, counts99 = _policy_surface(data99, "visual_stock_delta")
    var95, _ = _policy_surface(data95, "visual_varswap_delta")
    var99, _ = _policy_surface(data99, "visual_varswap_delta")
    stock_surfaces = [
        ("Model call-spread stock delta", model_stock),
        ("Policy stock delta, CVaR 95%", stock95),
        ("Policy stock delta, CVaR 99%", stock99),
        ("Policy difference: 99% - 95%", stock99 - stock95),
    ]
    var_surfaces = [
        ("Model call-spread variance-swap delta", model_varswap),
        ("Policy variance-swap delta, CVaR 95%", var95),
        ("Policy variance-swap delta, CVaR 99%", var99),
        ("Policy difference: 99% - 95%", var99 - var95),
    ]
    _plot_3d(
        figure_root / "fig08_e2cs_stock_position.png",
        "E2 Call Spread: Stock Hedge Surface", sg, vg, stock_surfaces,
    )
    _plot_3d(
        figure_root / "fig09_e2cs_varswap_position.png",
        "E2 Call Spread: Variance-Swap Hedge Surface", sg, vg, var_surfaces,
    )
    _plot_heatmaps(
        figure_root / "fig08_e2cs_stock_position_heatmap.png",
        "E2 Call Spread: Stock Hedge Heatmaps", sg, vg, stock_surfaces,
    )
    _plot_heatmaps(
        figure_root / "fig09_e2cs_varswap_position_heatmap.png",
        "E2 Call Spread: Variance-Swap Hedge Heatmaps", sg, vg, var_surfaces,
    )
    count_path = figure_root / "e2cs_surface_bin_counts.npz"
    np.savez_compressed(
        count_path,
        counts_alpha95=counts95,
        counts_alpha99=counts99,
        spot_grid=sg,
        variance_grid=vg,
    )
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    mesh = ax.pcolormesh(sg, vg, counts95.T, shading="auto", cmap="magma")
    ax.set_xlabel("Spot price S")
    ax.set_ylabel("Variance V")
    ax.set_title("E2CS visualisation sample count per (S,V) bin")
    fig.colorbar(mesh, ax=ax, label="Number of paths")
    fig.savefig(figure_root / "fig08_e2cs_surface_bin_counts.png", dpi=220)
    plt.close(fig)
    print(f"Figures saved under {figure_root}; bin counts saved to {count_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
