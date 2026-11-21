"""Run the paired E2 call-spread experiment and write per-alpha result folders."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from src.common.config import dump_runtime_config, load_config
from src.common.logging import human_secs, setup_logger
from src.common.metrics import save_pnl, write_metrics
from src.common.seed import resolve_device
from src.replication.e2cs import run_pair


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp/e2cs.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    device = resolve_device(cfg["device"])
    root = Path(args.out)
    folders = [root / "e2cs" / alpha / str(args.seed) for alpha in ("0.95", "0.99")]
    if args.resume and all((folder / "metrics.json").exists() for folder in folders):
        print("E2CS seed already has both alpha outputs; skipping.")
        return 0
    if any((folder / "metrics.json").exists() for folder in folders):
        raise FileExistsError(
            "Partial E2CS outputs exist. Move them or use a fresh --out directory."
        )

    master_log = root / "e2cs" / f"seed_{args.seed}_master.log"
    master_log.parent.mkdir(parents=True, exist_ok=True)
    log = setup_logger(f"a2.e2cs.{args.seed}", master_log)
    started = time.perf_counter()
    log.info("E2CS seed=%d device=%s config=%s", args.seed, device, args.config)
    result = run_pair(cfg, args.seed, device, log)
    for alpha, payload in result["payloads"].items():
        folder = root / "e2cs" / alpha / str(args.seed)
        folder.mkdir(parents=True, exist_ok=True)
        local_cfg = dict(cfg)
        local_cfg["alpha"] = float(alpha)
        local_cfg["payoff"] = dict(cfg.get("payoff", {}))
        dump_runtime_config(local_cfg, folder / "config.yaml")
        save_pnl(folder / "pnl.npz", payload["terminal_loss"], **payload["extra_pnl"])
        torch.save(payload["policy"], folder / "policy.pt")
        write_metrics(
            folder / "metrics.json", exp="e2cs", seed=args.seed, cfg=local_cfg,
            metrics=result["metrics"], checks=result["checks"],
            runtime_sec=time.perf_counter() - started, device=device, n_paths=result["n_paths"],
        )
        shutil.copy2(master_log, folder / "run.log")
    summary = {
        "exp": "e2cs", "seed": args.seed, "device": str(device),
        "n_paths": result["n_paths"], "metrics": result["metrics"],
        "checks": result["checks"], "runtime_sec": time.perf_counter() - started,
    }
    summary_path = root / "e2cs" / f"seed_{args.seed}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info("E2CS completed in %s", human_secs(time.perf_counter() - started))
    for folder in folders:
        shutil.copy2(master_log, folder / "run.log")
    print(f"E2CS outputs: {folders[0]} and {folders[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
