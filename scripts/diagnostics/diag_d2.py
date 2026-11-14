"""d=2 with layer norm: 单 seed 快速验证（临时脚本）。"""
import logging, torch, yaml
from src.replication.e1 import market_from_cfg, policy_from_cfg, train_policy, delta_rmse
from src.replication.env_heston import HestonWorld
from src.replication.benchmark import HestonBenchmark
from src.common.metrics import cvar

cfg = yaml.safe_load(open("configs/exp/e1.yaml", encoding="utf-8"))
cfg["policy"]["norm"] = "layer"
torch.manual_seed(0)
world = HestonWorld(market_from_cfg(cfg))
pol = train_policy(world, cfg, torch.device("cpu"), logging.getLogger("x"))
torch.manual_seed(10000)
paths = world.simulate(100_000)
with torch.no_grad():
    pos = pol.eval().forward_path(world.features(paths))
loss = world.terminal_loss(pos, paths)
bm = HestonBenchmark(world)
rmse, mae = delta_rmse(paths, pos, bm)
bmloss = world.terminal_loss(bm.positions(paths), paths)
print("d2 layer: price %.4f mean %.4f cvar99 %.3f delta_rmse %.4f bm_price %.4f bm_mean %.4f"
      % (cvar(loss, 0.5), loss.mean(), cvar(loss, 0.99), rmse, cvar(bmloss, 0.5), bmloss.mean()))
