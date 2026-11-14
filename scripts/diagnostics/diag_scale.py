"""d=2 out_scale 试验（临时脚本）。"""
import logging, torch, yaml, time
from src.replication.e1 import market_from_cfg, policy_from_cfg
from src.replication.env_heston import HestonWorld
from src.replication.benchmark import HestonBenchmark
from src.replication.losses import avg_cvar
from src.common.metrics import cvar

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("diag")
cfg = yaml.safe_load(open("configs/exp/e1.yaml", encoding="utf-8"))
cfg["policy"]["norm"] = "layer"
cfg["policy"]["out_scale"] = [1.0, 100.0]
world = HestonWorld(market_from_cfg(cfg)); m = world.market
bm = HestonBenchmark(world)
alpha = float(cfg["alpha"]); batch = 256
torch.manual_seed(0)
pol = policy_from_cfg(cfg, m.n_features, m.d, torch.device("cpu"))
opt = torch.optim.Adam(pol.parameters(), lr=float(cfg["train"]["lr"]))
t0 = time.time()
for it in range(1, 20001):
    pol.train()
    paths = world.simulate(batch)
    pos = pol.forward_path(world.features(paths))
    loss = avg_cvar(world.terminal_loss(pos, paths), alpha)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if it % 2500 == 0:
        pol.eval(); torch.manual_seed(777)
        ev = world.simulate(20000)
        with torch.no_grad():
            p = pol.forward_path(world.features(ev))
        l = world.terminal_loss(p, ev); bl = world.terminal_loss(bm.positions(ev), ev)
        k = 15
        b1, b2 = bm.delta(ev.spot[:, k].numpy(), ev.variance[:, k].numpy(), k)
        log.info("iter %5d %.0fs price=%.4f (bm %.4f) d1=%.2f/%.2f d2=%.0f/%.0f",
                 it, time.time() - t0, cvar(l, 0.5), cvar(bl, 0.5),
                 p[:, k, 0].mean(), b1.mean(), p[:, k, 1].mean(), b2.mean())