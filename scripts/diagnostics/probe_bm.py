import torch, numpy as np
from src.common.seed import seed_everything
from src.common.metrics import cvar
from src.replication.e1 import HestonMarket, HestonWorld
from src.replication.benchmark import HestonBenchmark
seed_everything(11)
w = HestonWorld(HestonMarket(d=2))
p = w.simulate(100000)
s2 = p.varswap
print("E[S2_0]", float(s2[:,0].mean()), "E[S2_T]", float(s2[:,-1].mean()), "se", float(s2[:,-1].std()/np.sqrt(len(s2))), "rel", float(s2[:,-1].mean()/s2[:,0].mean()-1))
bm = HestonBenchmark(w)
pos = bm.positions(p)
L = w.terminal_loss(pos, p)
print("bm price", round(float(cvar(L,0.5)),4), "mean", round(float(L.mean()),4), "cvar99", round(float(cvar(L,0.99)),4))
d1 = pos[:,0,0].mean(); d2 = pos[:,0,1].mean(); d2b = pos[:,15,1].mean()
print("bm d1 k0", round(float(d1),4), "d2 k0", round(float(d2),1), "d2 k15", round(float(d2b),1))
print("loss min/max", round(float(L.min()),3), round(float(L.max()),3))
