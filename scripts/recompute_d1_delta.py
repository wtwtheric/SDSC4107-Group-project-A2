"""重算 E1d1 的 δ1 RMSE / MAE（模型 δ1 对 Heston 解析 δ1，k=15，S∈[90,110]）。

用来复核 method_replication §9.2 里「δ1 RMSE 0.24（MAE 0.19）」这句：`pnl.npz` 只存 loss，
不存 delta，所以要单独前向一次。

用法：
    python -m scripts.recompute_d1_delta                 # seed 0，20000 条新路径
    python -m scripts.recompute_d1_delta --seed 3 --paths 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from src.replication.benchmark import HestonBenchmark
from src.replication.e1 import delta_rmse, evaluate, market_from_cfg, policy_from_cfg
from src.replication.env_heston import HestonWorld


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="e1d1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paths", type=int, default=20_000)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--lo", type=float, default=90.0)
    ap.add_argument("--hi", type=float, default=110.0)
    args = ap.parse_args(argv)

    run = Path("results") / args.exp / str(args.seed)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    market = market_from_cfg(cfg)
    world = HestonWorld(market, device=torch.device("cpu"))
    policy = policy_from_cfg(cfg, market.n_features, market.d, torch.device("cpu"))
    policy.load_state_dict(torch.load(run / "policy.pt", map_location="cpu", weights_only=True))

    offset = int((cfg.get("train") or {}).get("eval_seed_offset", 10_000))
    torch.manual_seed(args.seed + offset)
    ev = evaluate(world, policy, args.paths)
    rmse, mae = delta_rmse(
        ev["paths"], ev["positions"], HestonBenchmark(world), k=args.k, lo=args.lo, hi=args.hi
    )
    spot = ev["paths"].spot[:, args.k]
    inside = int(((spot >= args.lo) & (spot <= args.hi)).sum())
    print(
        f"exp={args.exp} seed={args.seed} paths={args.paths} k={args.k} "
        f"S=[{args.lo:g},{args.hi:g}] inside={inside}/{args.paths} "
        f"delta_rmse={rmse:.4f} delta_mae={mae:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
