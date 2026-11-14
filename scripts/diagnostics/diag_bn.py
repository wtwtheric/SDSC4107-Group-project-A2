import torch, yaml
from src.replication.e1 import market_from_cfg, policy_from_cfg
from src.replication.env_heston import HestonWorld

cfg = yaml.safe_load(open("configs/exp/e1d1.yaml", encoding="utf-8"))
world = HestonWorld(market_from_cfg(cfg))
pol = policy_from_cfg(cfg, world.market.n_features, world.market.d, torch.device("cpu"))
pol.load_state_dict(torch.load("results/e1d1/3/policy.pt", map_location="cpu")); pol = pol.double(); pol.eval()
torch.manual_seed(10003)
paths = world.simulate(100_000)
with torch.no_grad():
    pos = pol.forward_path(world.features(paths))
loss = world.terminal_loss(pos, paths)
print("price %.4f mean %.4f" % (torch.quantile(loss, 0.5) * 0 + (-torch.sort(loss).values[:50000].mean()), loss.mean()))
for q in (0.5, 0.9, 0.99, 0.999, 0.9999):
    print("q%.4f: %.3f" % (q, torch.quantile(loss, q)))
print("max|delta| k=0,1,2,15,29:", [round(float(pos[:, k].abs().max()), 2) for k in (0, 1, 2, 15, 29)])
bad = loss > 50
print("n paths loss>50:", int(bad.sum()), "of", loss.numel())
if int(bad.sum()):
    idx = torch.nonzero(bad).flatten()[:5]
    for i in idx.tolist():
        print("path", i, "S0..S5", [round(float(x),1) for x in paths.spot[i, :6]], "vmax", round(float(paths.variance[i].max()),3),
              "pos k0..2", [round(float(x),1) for x in pos[i, :3]], "loss", round(float(loss[i]),1))
