"""把 e1d1 的某个 seed（默认 0）冻结成 src/policy/weights/heston_d1.pt，供 B2 回测。

用法：
    python -m scripts.export_policy_d1            # 用 seed 0
    python -m scripts.export_policy_d1 --seed 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from src.common.metrics import git_provenance
from src.policy.heston_d1 import WEIGHTS
from src.replication.e1 import market_from_cfg, policy_from_cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="e1d1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(WEIGHTS))
    args = ap.parse_args(argv)

    run = Path("results") / args.exp / str(args.seed)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))

    market = market_from_cfg(cfg)
    policy = policy_from_cfg(cfg, market.n_features, market.d, torch.device("cpu"))
    policy.load_state_dict(torch.load(run / "policy.pt", map_location="cpu", weights_only=True))
    meta = {
        "policy": "policy_heston_d1",
        "source_run": str(run).replace("\\", "/"),
        "seed": args.seed,
        "n_features": market.n_features,
        "d": market.d,
        "n_steps": market.n_steps,
        "hidden": int(cfg["policy"].get("hidden", market.d + 15)),
        "recurrent": bool(cfg["policy"].get("recurrent", True)),
        "batch_norm": bool(cfg["policy"].get("batch_norm", True)),
        "norm": cfg["policy"].get("norm"),
        "out_scale": cfg["policy"].get("out_scale", 1.0),
        "feature_spec": ["log(S_k/S_0)", "t_k/T"],
        "market": {
            k: cfg["market"][k] for k in ("s0", "k", "t", "kappa", "theta", "sigma", "rho", "v0")
        },
        "training": cfg["train"],
        "eval_metrics": metrics["metrics"],
        "source_git_commit": metrics.get("git_commit"),
        "export_git_commit": git_provenance()["git_commit"],
        "dtype": "float64",
        "doc": "d=1 Heston deep hedge; input x_k = [log(S_k/S_0), t_k/T]; output = units of the underlying.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "meta": meta}, out)
    print(f"exported {out} ({out.stat().st_size / 1024:.0f} KB) from {run}")
    print(json.dumps(meta["eval_metrics"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
