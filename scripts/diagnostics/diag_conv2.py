"""一次性收敛试验：batch 1024 + cosine lr（用完即删）"""
from __future__ import annotations
import sys, time, math
import torch
from src.common.seed import seed_everything
from src.common.metrics import cvar
from src.replication.e1 import HestonMarket, HestonWorld, policy_from_cfg, q_price
from src.replication.losses import avg_cvar
from src.replication.benchmark import HestonBenchmark

mode = sys.argv[1] if len(sys.argv) > 1 else "varswap"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
LR0, LR1 = 5e-3, 5e-4
BATCH = 1024
dev = torch.device("cpu")
market = HestonMarket(d=2)
world = HestonWorld(market, device=dev)
b = market.theta

def feats(paths):
    n = market.n_steps
    ls = torch.log(paths.spot[:, :n] / market.s0)
    t = (paths.time[:n] / market.maturity).unsqueeze(0).expand(paths.spot.shape[0], n)
    s2 = paths.varswap[:, :n] / paths.varswap[:, :1] - 1.0
    v = paths.variance[:, :n] / b - 1.0
    if mode == "var":
        return torch.stack([ls, t, v], -1)
    if mode == "both":
        return torch.stack([ls, t, v, s2], -1)
    return torch.stack([ls, t, s2], -1)

seed_everything(0)
nf = 3 if mode != "both" else 4
pol = policy_from_cfg({"policy": {"hidden": 17, "norm": "layer", "out_scale": 100.0}, "n_steps": 30}, nf, 2, dev)
opt = torch.optim.Adam(pol.parameters(), lr=LR0)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N, eta_min=LR1)
t0 = time.time()
for it in range(1, N + 1):
    pol.train()
    paths = world.simulate(BATCH)
    loss = avg_cvar(world.terminal_loss(pol.forward_path(feats(paths)), paths), 0.5)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    if it % 5000 == 0:
        print(f"  [{mode}] iter {it} {time.time()-t0:.0f}s cvar={float(loss.detach()):.4f}", flush=True)

pol.eval()
torch.manual_seed(999)
with torch.no_grad():
    paths = world.simulate(100_000)
    pos = pol.forward_path(feats(paths))
    L = world.terminal_loss(pos, paths)
    bm = HestonBenchmark(world)
    bL = world.terminal_loss(bm.positions(paths), paths)
print(f"[{mode}] price={float(cvar(L,0.5)):.4f} mean={float(L.mean()):.4f} q={q_price(world):.4f} "
      f"cvar99={float(cvar(L,0.99)):.3f} bm={float(cvar(bL,0.5)):.4f} "
      f"d1={[round(float(pos[:,k,0].mean()),3) for k in (0,15,29)]} "
      f"d2={[round(float(pos[:,k,1].mean()),0) for k in (0,15,29)]}", flush=True)
torch.save(pol.state_dict(), f"results/_diag_{mode}.pt")
