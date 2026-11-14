"""一次性：拆解 E1 单次迭代耗时（simulate / features / forward / loss / backward）。"""

from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, ".")

from src.common.config import load_config
from src.replication import e1
from src.replication.losses import avg_cvar


def timer(fn, n=20):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e3


def main() -> None:
    cfg = load_config("configs/exp/e1.yaml")
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else int(cfg["batch"])
    device = torch.device("cpu")
    market = e1.market_from_cfg(cfg)
    world = e1.HestonWorld(market, device=device)
    policy = e1.policy_from_cfg(cfg, market.n_features, market.d, device)
    opt = torch.optim.Adam(policy.parameters(), lr=0.005)

    paths = world.simulate(batch)
    feats = world.features(paths)
    print(f"batch={batch} d={market.d} f={market.n_features} hidden={cfg['policy']['hidden']}")
    print(f"params={sum(p.numel() for p in policy.parameters())}")

    t_sim = timer(lambda: world.simulate(batch))
    t_feat = timer(lambda: world.features(paths))
    t_fwd = timer(lambda: policy.forward_path(feats))
    pos = policy.forward_path(feats)
    t_loss = timer(lambda: world.terminal_loss(pos, paths))
    loss = avg_cvar(world.terminal_loss(pos, paths), float(cfg["alpha"]))

    def full():
        p = world.simulate(batch)
        f = world.features(p)
        x = policy.forward_path(f)
        l = avg_cvar(world.terminal_loss(x, p), float(cfg["alpha"]))
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()

    print(f"simulate     {t_sim:8.3f} ms")
    print(f"features     {t_feat:8.3f} ms")
    print(f"forward_path {t_fwd:8.3f} ms")
    print(f"terminal_loss{t_loss:8.3f} ms")
    t_full = timer(full, n=30)
    print(f"full step    {t_full:8.3f} ms  -> 20k iter = {t_full * 20000 / 60:6.2f} min")


if __name__ == "__main__":
    main()
