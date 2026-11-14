import time, torch, yaml
from src.replication.e1 import market_from_cfg, policy_from_cfg
from src.replication.env_heston import HestonWorld
from src.replication.pricing import heston_call_prices
from src.replication.benchmark import HestonBenchmark
import numpy as np

cfg = yaml.safe_load(open("configs/exp/e1.yaml", encoding="utf-8"))
world = HestonWorld(market_from_cfg(cfg))
pol = policy_from_cfg(cfg, world.market.n_features, world.market.d, torch.device("cpu"))
# 计时
t = time.time(); [world.simulate(256) for _ in range(20)]; sim = (time.time() - t) / 20
paths = world.simulate(256); feats = world.features(paths)
t = time.time()
for _ in range(20):
    with torch.no_grad():
        pol.forward_path(feats)
fwd = (time.time() - t) / 20
print("simulate/iter %.4fs (%.0f%%)   forward/iter %.4fs (%.0f%%)" % (sim, 100 * sim / (sim + fwd), fwd, 100 * fwd / (sim + fwd)))
# benchmark delta1 有限差分校验（t=0）
bm = HestonBenchmark(world)
d1, d2 = bm.delta(torch.tensor([100.0]), torch.tensor([0.04]), 0)
h = 0.01
up = heston_call_prices(np.array([100 + h]), np.array([0.04]), 30 / 365)[0]
dn = heston_call_prices(np.array([100 - h]), np.array([0.04]), 30 / 365)[0]
print("benchmark delta1 k=0 %.4f ; finite diff %.4f" % (float(d1[0]), float((up - dn) / (2 * h))))
print("C(S=100)=%.4f C(110)=%.4f C(90)=%.4f" % (
    heston_call_prices(np.array([100.0]), np.array([0.04]), 30/365)[0],
    heston_call_prices(np.array([110.0]), np.array([0.04]), 30/365)[0],
    heston_call_prices(np.array([90.0]), np.array([0.04]), 30/365)[0]))
