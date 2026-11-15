"""Create paper-style E2CS stock-hedge Figures 8 and 9.

Figure 8 is the model hedge vs alpha=.95 simpler NN hedge.
Figure 9 is the model hedge vs alpha=.99 simpler NN hedge.
The NN surfaces are evaluated directly on (S,V), not interpolated from paths.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def _three_surfaces(data: dict[str, np.ndarray], alpha: str):
    s = data["model_spot_grid"].astype(float)
    v = data["model_variance_grid"].astype(float)
    model = data["model_stock_delta_grid"].astype(float)
    policy = data["policy_stock_delta_grid"].astype(float)
    if model.shape != policy.shape or model.shape != (len(s), len(v)):
        raise ValueError(f"Unexpected surface shapes: model={model.shape}, policy={policy.shape}")
    if not (np.isfinite(model).all() and np.isfinite(policy).all()):
        raise ValueError("Non-finite values in E2CS stock hedge surface")
    return s, v, [
        ("Model hedge", model),
        (f"Neural-network hedge (CVaR {alpha}%)", policy),
        ("Difference: NN − model", policy - model),
    ]


def _symmetric_norm(values: list[np.ndarray]) -> TwoSlopeNorm:
    limit = max(float(np.nanmax(np.abs(x))) for x in values)
    limit = max(limit, 1.0e-12)
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def _plot_3d(out: Path, alpha: str, s: np.ndarray, v: np.ndarray, surfaces) -> None:
    ss, vv = np.meshgrid(s, v, indexing="ij")
    comparable_norm = _symmetric_norm([surfaces[0][1], surfaces[1][1]])
    difference_norm = _symmetric_norm([surfaces[2][1]])
    fig = plt.figure(figsize=(15, 5.4), constrained_layout=True)
    for idx, (label, z) in enumerate(surfaces, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        norm = difference_norm if idx == 3 else comparable_norm
        ax.plot_surface(
            ss, vv, z, cmap="coolwarm", norm=norm,
            linewidth=0, antialiased=True, rstride=1, cstride=1,
        )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Spot price S")
        ax.set_ylabel("Variance V")
        ax.set_zlabel("Stock hedge position")
        ax.set_xlim(float(s[0]), float(s[-1]))
        ax.set_ylim(float(v[0]), float(v[-1]))
        ax.view_init(elev=28, azim=-45)
    fig.suptitle(
        f"Call-spread stock hedge at k=15 — CVaR {alpha}%",
        fontsize=14,
    )
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmaps(out: Path, alpha: str, s: np.ndarray, v: np.ndarray, surfaces) -> None:
    comparable_norm = _symmetric_norm([surfaces[0][1], surfaces[1][1]])
    difference_norm = _symmetric_norm([surfaces[2][1]])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for idx, (ax, (label, z)) in enumerate(zip(axes, surfaces, strict=True), start=1):
        norm = difference_norm if idx == 3 else comparable_norm
        mesh = ax.pcolormesh(s, v, z.T, shading="auto", cmap="coolwarm", norm=norm)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Spot price S")
        ax.set_ylabel("Variance V")
        fig.colorbar(mesh, ax=ax, label="Stock hedge position")
    fig.suptitle(f"Call-spread stock hedge at k=15 — CVaR {alpha}%", fontsize=14)
    fig.savefig(out, dpi=300, bbox_inches="tight")
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

    for alpha, number in (("0.95", "08"), ("0.99", "09")):
        path = result_root / alpha / str(args.seed) / "pnl.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing E2CS result for alpha={alpha}: {path}")
        with np.load(path) as archive:
            data = {key: archive[key] for key in archive.files}
        s, v, surfaces = _three_surfaces(data, str(int(float(alpha) * 100)))
        _plot_3d(
            figure_root / f"fig{number}_e2cs_stock_position.png",
            str(int(float(alpha) * 100)), s, v, surfaces,
        )
        _plot_heatmaps(
            figure_root / f"fig{number}_e2cs_stock_position_heatmap.png",
            str(int(float(alpha) * 100)), s, v, surfaces,
        )

    # Keep a separate path-support diagnostic; it does not define/interpolate the NN surface.
    with np.load(result_root / "0.95" / str(args.seed) / "pnl.npz") as data95:
        counts, _, _ = np.histogram2d(
            data95["visual_spot"], data95["visual_variance"],
            bins=[data95["model_spot_grid"], data95["model_variance_grid"]],
        )
        s = data95["model_spot_grid"]
        v = data95["model_variance_grid"]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    mesh = ax.pcolormesh(s, v, counts.T, shading="auto", cmap="magma")
    ax.set_xlabel("Spot price S")
    ax.set_ylabel("Variance V")
    ax.set_title("E2CS visualisation path support at k=15")
    fig.colorbar(mesh, ax=ax, label="Number of paths")
    fig.savefig(figure_root / "fig08_e2cs_surface_bin_counts.png", dpi=300)
    plt.close(fig)
    print(f"Paper-style stock hedge Figures 8 and 9 saved under {figure_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
