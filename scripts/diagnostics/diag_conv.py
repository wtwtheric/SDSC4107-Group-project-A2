"""d=2 收敛诊断：训练中定期评估 price 与 δ 的量级（临时脚本）。"""
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
alpha = float(cfg["alpha"]); batch = 256
world = HestonWorld(market_from_cfg(cfg))
m = world.market
bm = HestonBenchmark(world)
b1, b2 = bm.delta(torch.tensor([100.0, 100.0]), torch.tensor([0.04, 0.04]), 0)
log.info("benchmark delta k=0: d1=%s d2=%s", b1, b2)
b1, b2 = bm.delta(torch.tensor([100.0, 100.0]), torch.tensor([0.04, 0.04]), 15)
log.info("benchmark delta k=15: d1=%s d2=%s", b1, b2)

torch.manual_seed(0)
pol = policy_from_cfg(cfg, m.n_features, m.d, torch.device("cpu"))
opt = torch.optim.Adam(pol.parameters(), lr=float(cfg["train"]["lr"]))
t0 = time.time()
for it in range(1, 15001):
    pol.train()
    paths = world.simulate(batch)
    pos = pol.forward_path(world.features(paths))
    loss = avg_cvar(world.terminal_loss(pos, paths), alpha)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if it % 3000 == 0:
        pol.eval()
        torch.manual_seed(777)
        ev = world.simulate(20000)
        with torch.no_grad():
            p_pos = pol.forward_path(world.features(ev))
        l = world.terminal_loss(p_pos, ev)
        bm_l = world.terminal_loss(bm.positions(ev), ev)
        k = 15
        s15 = ev.spot[:, k]
        d1m, _ = bm.delta(s15.numpy(), ev.variance[:, k].numpy() if hasattr(ev.variance[:, k], "numpy") else ev.variance[:, k], k)
        log.info("iter %5d  %.0fs  price=%.4f (bm %.4f)  mean_d1=%.3f mean_d2=%.1f  bm_d1=%.3f bm_d2=%.1f",
                 it, time.time() - t0, cvar(l, 0.5), cvar(bm_l, 0.5),
                 p_pos[:, k, 0].mean(), p_pos[:, k, 1].mean(), d1m.mean(),
                 float(bm.delta(s15, ev.variance[:, k], k)[1].mean()))
