"""快速对照：同一 seed 下 batch-norm vs layer-norm 的 eval 稳定性（临时脚本）。"""
import sys, torch, yaml
from src.replication.e1 import market_from_cfg, policy_from_cfg, train_policy
from src.replication.env_heston import HestonWorld
from src.common.metrics import cvar

kind = sys.argv[1]
cfg = yaml.safe_load(open("configs/exp/e1d1.yaml", encoding="utf-8"))
cfg["policy"]["norm"] = kind
cfg["train"]["n_iter"] = int(cfg["train"].get("n_iter", 3000))
torch.manual_seed(0)
world = HestonWorld(market_from_cfg(cfg))
pol = train_policy(world, cfg, torch.device("cpu"), __import__("logging").getLogger("x"))
torch.manual_seed(10000)
paths = world.simulate(100_000)
with torch.no_grad():
    pos = pol.eval().forward_path(world.features(paths))
loss = world.terminal_loss(pos, paths)
print(kind, "price %.4f mean %.4f cvar99 %.3f  max|delta| %.2f  min/max loss %.2f/%.2f"
      % (cvar(loss, 0.5), loss.mean(), cvar(loss, 0.99), pos.abs().max(), loss.min(), loss.max()))
