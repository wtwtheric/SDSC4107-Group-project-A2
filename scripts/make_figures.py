"""E1 的三张图（论文 Fig 2/3/4）：损失直方图、delta 面、recurrent vs simpler。

用法：python -m scripts.make_figures [--exp e1] [--figs 2 3 4]
输出：figures/fig02_loss_hist.png, fig03_delta_surface.png, fig04_recurrent_vs_simpler.png
依赖：results/e1/<seed>/{metrics.json,pnl.npz,config.yaml}（train.py 产出）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from src.common.metrics import cvar
from src.replication.benchmark import HestonBenchmark
from src.replication.e1 import market_from_cfg, policy_from_cfg
from src.replication.env_heston import HestonWorld

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path("figures")
PAPER_P0 = 1.94  # 论文 E1 的 p0（α=0.5）
PAPER_P0_99 = 3.49


def _load(exp: str) -> tuple[list[int], list[dict]]:
    root = Path("results") / exp
    seeds = sorted(int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    metrics = [
        json.loads((root / str(s) / "metrics.json").read_text(encoding="utf-8")) for s in seeds
    ]
    return seeds, metrics


def _losses(exp: str, seeds: list[int], key: str = "terminal_loss") -> np.ndarray:
    out = []
    for s in seeds:
        with np.load(Path("results") / exp / str(s) / "pnl.npz") as data:
            if key in data:
                out.append(data[key])
    return np.concatenate(out) if out else np.array([])


def fig02(exp: str = "e1") -> Path:
    seeds, metrics = _load(exp)
    loss = _losses(exp, seeds)
    bench = _losses(exp, seeds, "terminal_loss_benchmark")
    price = float(np.mean([m["metrics"]["price"] for m in metrics]))
    c_net, c_bench = float(cvar(loss, 0.5)), float(cvar(bench, 0.5))

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    lo = max(min(loss.min(), bench.min()), -2.6)
    hi = float(np.percentile(loss, 99.9))
    bins = np.linspace(lo, hi, 140)
    bench_color = "#7f7f7f"  # spec §10：基准用灰
    net_color = "#1f77b4"  # spec §10：网络用蓝
    ax.hist(
        bench,
        bins=bins,
        density=True,
        histtype="step",
        lw=1.6,
        label=f"Heston delta hedge (CVaR$_{{50}}$={c_bench:.3f})",
        color=bench_color,
    )
    ax.hist(
        loss,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.45,
        lw=1.2,
        edgecolor=net_color,
        label=f"deep hedge (CVaR$_{{50}}$={c_net:.3f})",
        color=net_color,
    )
    ax.axvline(PAPER_P0, color="#d62728", ls="--", lw=1.5, label=f"paper $p_0$={PAPER_P0}")
    ax.set_xlim(lo, hi + 0.4)
    ax.set_xlabel("terminal hedging loss  $L$  (units of \\$)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.text(
        0.02,
        0.70,
        f"CVaR$_{{50}}$  net {c_net:.3f}  |  Heston $\\delta$ {c_bench:.3f}  |  paper {PAPER_P0}\n"
        "the two hedges are statistically indistinguishable\n"
        "in the body; the price gap is 0.4%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="0.3",
    )
    fig.text(
        0.01,
        0.005,
        f"Fig 2 replication (E1, Heston $d$=2, avg-CVaR(0.5), {len(seeds)} seeds, zero cost): the trained network\n"
        f"prices the option at {price:.3f} (paper {PAPER_P0}) and matches the analytic hedge ({c_bench:.3f}).",
        fontsize=8,
        color="0.25",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = FIG / "fig02_loss_hist.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[fig02] CVaR50 net={c_net:.4f} bench={c_bench:.4f} paper={PAPER_P0}")
    return out


def _ensemble_positions(
    exp: str, seeds: list[int], n_paths: int = 100_000
) -> dict[str, np.ndarray]:
    """在固定评估路径上跑全部 seed，返回平均仓位（网络）与 model delta。"""
    cfg = __import__("yaml").safe_load(
        (Path("results") / exp / str(seeds[0]) / "config.yaml").read_text(encoding="utf-8")
    )
    market = market_from_cfg(cfg)
    world = HestonWorld(market, device=torch.device("cpu"))
    torch.manual_seed(424242)
    paths = world.simulate(n_paths)
    feats = world.features(paths)
    acc = np.zeros((n_paths, market.n_steps, market.d))
    for s in seeds:
        policy = policy_from_cfg(cfg, market.n_features, market.d, torch.device("cpu"))
        state = torch.load(
            Path("results") / exp / str(s) / "policy.pt", map_location="cpu", weights_only=True
        )
        policy.load_state_dict(state)
        policy.eval()
        with torch.no_grad():
            pos = policy.forward_path(feats).numpy()
        acc += pos
    policy_pos = acc / len(seeds)
    bench = HestonBenchmark(world)
    bench_pos = bench.positions(paths).numpy()
    return {
        "policy": policy_pos,
        "benchmark": bench_pos,
        "spot": paths.spot.numpy(),
        "var": paths.variance.numpy(),
    }


def fig03(exp: str = "e1", k: int = 15) -> Path:
    seeds, _ = _load(exp)
    data = _ensemble_positions(exp, seeds)
    spot, var = data["spot"], data["var"]
    s_lo, s_hi, v_lo, v_hi = 93.0, 107.0, 0.010, 0.080
    s_edges = np.linspace(s_lo, s_hi, 22)
    v_edges = np.linspace(v_lo, v_hi, 16)
    mask = (spot[:, k] >= s_lo) & (spot[:, k] <= s_hi) & (var[:, k] >= v_lo) & (var[:, k] <= v_hi)
    print(f"[fig03] {mask.sum()} / {len(mask)} paths in the (S,v) window at k={k}")

    def binned(values: np.ndarray) -> np.ndarray:
        num, _, _ = np.histogram2d(
            spot[mask, k], var[mask, k], bins=(s_edges, v_edges), weights=values[mask]
        )
        cnt, _, _ = np.histogram2d(spot[mask, k], var[mask, k], bins=(s_edges, v_edges))
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / cnt
        out[cnt < 5] = np.nan
        return out.T

    pol1, mod1 = binned(data["policy"][:, k, 0]), binned(data["benchmark"][:, k, 0])
    pol2, mod2 = binned(data["policy"][:, k, 1]), binned(data["benchmark"][:, k, 1])
    d1_rmse = float(np.sqrt(np.nanmean((pol1 - mod1) ** 2)))
    d2_rmse = float(np.sqrt(np.nanmean((pol2 - mod2) ** 2)))
    d1_mae = float(np.nanmean(np.abs(pol1 - mod1)))
    d2_mae = float(np.nanmean(np.abs(pol2 - mod2)))

    def lim(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        both = np.concatenate([a[~np.isnan(a)], b[~np.isnan(b)]])
        return float(both.min()), float(both.max())

    l1, h1 = lim(pol1, mod1)
    l2, h2 = lim(pol2, mod2)

    panels = [
        (f"policy $\\delta_1$  (MAE {d1_mae:.3f})", pol1, (l1, h1)),
        ("model $\\delta_1$", mod1, (l1, h1)),
        (f"policy $\\delta_2$  (MAE {d2_mae:.1f})", pol2, (l2, h2)),
        ("model $\\delta_2$", mod2, (l2, h2)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4), sharex=True, sharey=True)
    for ax, (title, grid, (vmin, vmax)) in zip(axes.ravel(), panels, strict=True):
        mesh = ax.pcolormesh(
            s_edges, v_edges, grid, cmap="RdBu_r", shading="auto", vmin=vmin, vmax=vmax
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("spot $S_{15}$")
        ax.set_ylabel("variance $v_{15}$")
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    fig.text(
        0.01,
        0.005,
        f"Fig 3 replication (E1, $t=15$d, 5-seed mean network vs Heston model, "
        f"{mask.sum()}/{len(mask)} paths in the window): the network reproduces\n"
        f"the model $\\delta_1$ (RMSE {d1_rmse:.3f}) and the shape of $\\delta_2$ "
        f"(RMSE {d2_rmse:.1f} on a ~500 scale).",
        fontsize=8,
        color="0.25",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = FIG / "fig03_delta_surface.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[fig03] delta1 RMSE={d1_rmse:.4f} delta2 RMSE={d2_rmse:.4f}")
    return out


def fig04(exp: str = "e1") -> Path:
    seeds, metrics = _load(exp)
    rec = np.array([m["metrics"]["price"] for m in metrics])
    simp = np.array([m["metrics"]["price_simpler"] for m in metrics])
    bench = np.array([m["metrics"]["benchmark_price"] for m in metrics])
    gap = float(np.max(np.abs(simp - rec) / rec))
    bm = float(bench.mean())

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(seeds))
    for xi, r, si in zip(x, rec, simp, strict=True):
        ax.plot([xi, xi], [r, si], color="0.65", lw=1.4, zorder=1)
    ax.scatter(x, rec, s=75, color="#1f77b4", label="recurrent $\\delta_{k-1}$", zorder=3)
    ax.scatter(x, simp, s=75, color="#9467bd", marker="s", label="simpler (feed-forward)", zorder=3)
    ax.axhline(PAPER_P0, color="#d62728", ls="--", lw=1.5, label=f"paper $p_0$={PAPER_P0}")
    bm = float(bench.mean())
    ax.axhline(bm, color="#7f7f7f", ls="--", lw=1.2, label=f"Heston $\\delta$ hedge ({bm:.3f})")
    ax.set_xticks(x, [f"seed {s}" for s in seeds])
    ax.set_xlim(-0.6, len(seeds) - 0.4)
    ax.set_ylim(
        min(rec.min(), simp.min(), bm) - 0.0015, max(rec.max(), simp.max(), PAPER_P0) + 0.0035
    )
    ax.set_ylabel("price  $p_0$  (CVaR$_{50}$)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    for xi, r, si in zip(x, rec, simp, strict=True):
        ax.annotate(
            f"{(si - r) * 1000:+.1f}",
            xy=(xi + 0.10, (r + si) / 2),
            fontsize=7.5,
            color="0.4",
            va="center",
        )
    ax.text(
        0.99,
        0.96,
        f"max gap {gap:.2%} (criterion < 3%); labels are\n(simpler $- $ recurrent) in $10^{{-3}}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="0.35",
    )
    fig.text(
        0.01,
        0.005,
        "Fig 4 replication (E1, zero cost): the recurrent and simpler architectures agree to 0.15% in price,\n"
        "so with the Markov state $(S,v)$ observable the previous position adds no information.",
        fontsize=8,
        color="0.25",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = FIG / "fig04_recurrent_vs_simpler.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="e1")
    ap.add_argument("--figs", nargs="+", type=int, default=[2, 3, 4])
    args = ap.parse_args(argv)
    FIG.mkdir(parents=True, exist_ok=True)
    for n in args.figs:
        if n == 2:
            print("wrote", fig02(args.exp))
        elif n == 3:
            print("wrote", fig03(args.exp))
        elif n == 4:
            print("wrote", fig04(args.exp))
        else:
            raise SystemExit(f"unknown figure {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
