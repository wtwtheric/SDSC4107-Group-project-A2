"""一次性验证脚本：d=2 训练 20k iter 后逐项体检（用完即删）。"""
from __future__ import annotations
import sys, time
import numpy as np
import torch
from src.common.seed import seed_everything
from src.common.metrics import cvar
from src.replication.e1 import HestonMarket, HestonWorld, policy_from_cfg, q_price
from src.replication.losses import avg_cvar
from src.replication.benchmark import HestonBenchmark

cfg = {
    "market": {"d": 2},
    "n_steps": 30,
    "batch": 256,
    "alpha": 0.5,
    "eps": 0.0,
    "policy": {"hidden": 17, "norm": "layer", "out_scale": 100.0},
    "train": {"n_iter": 20000, "lr": 0.005, "log_every": 2500},
}
seed_everything(0)
dev = torch.device("cpu")
market = HestonMarket(d=2)
world = HestonWorld(market, device=dev)
pol = policy_from_cfg(cfg, market.n_features, market.d, dev)
opt = torch.optim.Adam(pol.parameters(), lr=0.005)
t0 = time.time()
for it in range(1, 20001):
    pol.train()
    paths = world.simulate(256)
    feats = world.features(paths)
    pos = pol.forward_path(feats)
    loss = avg_cvar(world.terminal_loss(pos, paths), 0.5)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if it % 2500 == 0:
        print(f"iter {it:6d} {time.time()-t0:6.0f}s train_cvar={float(loss):.4f}")

pol.eval()
torch.manual_seed(777)
with torch.no_grad():
    paths = world.simulate(100_000)
    feats = world.features(paths)
    pos = pol.forward_path(feats)
    L = world.terminal_loss(pos, paths)
    prices = world.prices(paths)
    step = (pos * (prices[:, 1:] - prices[:, :-1])).sum(-1)
    hedge = step.sum(-1)
    leg1 = (pos[..., 0] * (prices[:, 1:, 0] - prices[:, :-1, 0])).sum(-1)
    leg2 = (pos[..., 1] * (prices[:, 1:, 1] - prices[:, :-1, 1])).sum(-1)
    q = q_price(world)
    bm = HestonBenchmark(world)
    bpos = bm.positions(paths)
    bL = world.terminal_loss(bpos, paths)

def qs(x, ql=(0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999)):
    return {f"q{p}": round(float(np.quantile(x, p)), 3) for p in ql}

print("q        ", round(q, 4))
print("price    ", round(float(cvar(L, 0.5)), 4))
print("mean_loss", round(float(L.mean()), 4), "  (should equal q)")
print("cvar99   ", round(float(cvar(L, 0.99)), 4))
print("loss qs  ", qs(L.numpy()))
print("hedge pnl mean", round(float(hedge.mean()), 5), " leg1", round(float(leg1.mean()), 5), " leg2", round(float(leg2.mean()), 5))
print("payoff mean", round(float(world.payoff(paths).mean()), 5))
print("d1 k=0/15/30", [round(float(pos[:, k, 0].mean()), 3) for k in (0, 15, 30)])
print("d1 sd  k=0/15 ", [round(float(pos[:, k, 0].std()), 3) for k in (0, 15)])
print("d2 k=0/15/30", [round(float(pos[:, k, 1].mean()), 1) for k in (0, 15, 30)])
print("d2 sd  k=0/15 ", [round(float(pos[:, k, 1].std()), 1) for k in (0, 15)])
print("bm price ", round(float(cvar(bL, 0.5)), 4), " bm mean", round(float(bL.mean()), 4))
bm1 = bpos[:, 0, 0].mean(); bm2 = bpos[:, 0, 1].mean(); bm2b = bpos[:, 15, 1].mean()
print("bm d1 k0", round(float(bm1), 3), " bm d2 k0", round(float(bm2), 1), " k15", round(float(bm2b), 1))
print("bm d1 k15", round(float(np.mean(bpos[:, 15, 0].numpy())), 3), " bm d2 mean", round(float(bpos[:, :, 1].mean()), 1), " bm d1 mean", round(float(bpos[:, :, 0].mean()), 3))
torch.save(pol.state_dict(), "results/_diag_val.pt")
print("saved results/_diag_val.pt")
