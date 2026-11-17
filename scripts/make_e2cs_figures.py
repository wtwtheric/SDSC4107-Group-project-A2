"""Render E2CS Figures 8–9 from saved surfaces, without changing trained policies.

The main renderer accepts raw model/policy stock-hedge grids. Different
experiments can supply those arrays from pnl.npz or saved simpler checkpoints.
Blue-white-red coloring preserves a white neutral difference at zero, rather
than a visually dominant grey middle. Model and NN share their color limits
across BOTH CVaR levels, so Figure 8/9 colors are directly comparable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm


def _three_surfaces(data: dict[str, np.ndarray], alpha: str):
    s = np.asarray(data["model_spot_grid"], dtype=float)
    v = np.asarray(data["model_variance_grid"], dtype=float)
    model = np.asarray(data["model_stock_delta_grid"], dtype=float)
    policy = np.asarray(data["policy_stock_delta_grid"], dtype=float)
    if model.shape != policy.shape or model.shape != (len(s), len(v)):
        raise ValueError(f"Unexpected surface shapes: model={model.shape}, policy={policy.shape}")
    if not (np.isfinite(model).all() and np.isfinite(policy).all()):
        raise ValueError("Non-finite values in E2CS stock hedge surface")
    return (
        s,
        v,
        [
            ("Model hedge", model),
            (f"Neural-network hedge (CVaR {alpha}%)", policy),
            ("Difference: NN − model", policy - model),
        ],
    )


def _symmetric_norm(values: list[np.ndarray]) -> TwoSlopeNorm:
    limit = max(float(np.max(np.abs(x))) for x in values)
    limit = max(limit, 1.0e-12)
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def _comparable_norm(values: list[np.ndarray]) -> Normalize:
    """Common color interval for model and neural hedge (not centered at zero)."""
    low = min(0.0, *(float(np.min(z)) for z in values))
    high = max(0.0, *(float(np.max(z)) for z in values))
    if high <= low:
        high = low + 1.0e-12
    return Normalize(vmin=low, vmax=high)


def _plot_3d(
    out: Path,
    alpha: str,
    s: np.ndarray,
    v: np.ndarray,
    surfaces,
    comparable_norm: Normalize,
    difference_norm: TwoSlopeNorm,
) -> None:
    ss, vv = np.meshgrid(s, v, indexing="ij")
    fig = plt.figure(figsize=(15, 5.4), constrained_layout=True)
    for idx, (label, z) in enumerate(surfaces, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        norm = difference_norm if idx == 3 else comparable_norm
        ax.plot_surface(
            ss,
            vv,
            z,
            cmap="bwr",
            norm=norm,
            linewidth=0,
            antialiased=True,
            shade=False,
            rstride=1,
            cstride=1,
        )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Spot price S")
        ax.set_ylabel("Variance V")
        ax.set_zlabel("Stock hedge position" if idx != 3 else "NN − model")
        ax.set_xlim(float(s[0]), float(s[-1]))
        ax.set_ylim(float(v[0]), float(v[-1]))
        ax.view_init(elev=28, azim=-45)
    fig.suptitle(f"Call-spread stock hedge at k=15 — CVaR {alpha}%", fontsize=14)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmaps(
    out: Path,
    alpha: str,
    s: np.ndarray,
    v: np.ndarray,
    surfaces,
    comparable_norm: Normalize,
    difference_norm: TwoSlopeNorm,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for idx, (ax, (label, z)) in enumerate(zip(axes, surfaces, strict=True), start=1):
        norm = difference_norm if idx == 3 else comparable_norm
        mesh = ax.pcolormesh(s, v, z.T, shading="auto", cmap="bwr", norm=norm)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Spot price S")
        ax.set_ylabel("Variance V")
        fig.colorbar(
            mesh, ax=ax, label="Stock hedge position" if idx != 3 else "NN − model"
        )
    fig.suptitle(f"Call-spread stock hedge at k=15 — CVaR {alpha}%", fontsize=14)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_surface_pairs(
    data_by_alpha: dict[str, dict[str, np.ndarray]], figure_root: Path
) -> list[Path]:
    """Use identical model/NN and difference color scales for alpha=.95 and .99."""
    if set(data_by_alpha) != {"0.95", "0.99"}:
        raise ValueError("Both 0.95 and 0.99 E2CS surfaces are required")
    surfaces = {
        alpha: _three_surfaces(data, str(int(float(alpha) * 100)))
        for alpha, data in data_by_alpha.items()
    }
    comparable_norm = _comparable_norm(
        [z for _s, _v, entries in surfaces.values() for _, z in entries[:2]]
    )
    difference_norm = _symmetric_norm(
        [entries[2][1] for _s, _v, entries in surfaces.values()]
    )
    result = []
    for alpha, number in (("0.95", "08"), ("0.99", "09")):
        s, v, entries = surfaces[alpha]
        path_3d = figure_root / f"fig{number}_e2cs_stock_position.png"
        path_heatmap = figure_root / f"fig{number}_e2cs_stock_position_heatmap.png"
        _plot_3d(
            path_3d, str(int(float(alpha) * 100)), s, v, entries,
            comparable_norm, difference_norm,
        )
        _plot_heatmaps(
            path_heatmap, str(int(float(alpha) * 100)), s, v, entries,
            comparable_norm, difference_norm,
        )
        result.extend([path_3d, path_heatmap])
    return result


def plot_path_support(data: dict[str, np.ndarray], figure_root: Path) -> Path:
    """Optional observed-path density, not a prerequisite for model surface plots."""
    counts, _, _ = np.histogram2d(
        data["visual_spot"],
        data["visual_variance"],
        bins=[data["model_spot_grid"], data["model_variance_grid"]],
    )
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    mesh = ax.pcolormesh(
        data["model_spot_grid"],
        data["model_variance_grid"],
        counts.T,
        shading="auto",
        cmap="magma",
    )
    ax.set_xlabel("Spot price S")
    ax.set_ylabel("Variance V")
    ax.set_title("E2CS visualisation path support at k=15")
    fig.colorbar(mesh, ax=ax, label="Number of paths")
    path = figure_root / "fig08_e2cs_surface_bin_counts.png"
    figure_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/e2cs")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result_root = Path(args.results)
    figure_root = Path(args.figures)
    data_by_alpha = {}
    for alpha in ("0.95", "0.99"):
        path = result_root / alpha / str(args.seed) / "pnl.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing E2CS result for alpha={alpha}: {path}")
        with np.load(path, allow_pickle=False) as archive:
            data_by_alpha[alpha] = {key: archive[key] for key in archive.files}
    paths = plot_surface_pairs(data_by_alpha, figure_root)
    paths.append(plot_path_support(data_by_alpha["0.95"], figure_root))
    for path in paths:
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
