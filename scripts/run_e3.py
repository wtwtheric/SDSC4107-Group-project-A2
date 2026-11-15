"""Run one E3 seed using the common A2 QE-M simulator and paired OOS paths."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from src.common.config import dump_runtime_config, load_config
from src.common.logging import human_secs, setup_logger
from src.common.metrics import save_pnl, write_metrics
from src.common.seed import resolve_device
from src.replication.e3 import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp/e3.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    device = resolve_device(cfg["device"])
    folder = Path(args.out) / "e3" / str(args.seed)
    if args.resume and (folder / "metrics.json").exists():
        print(f"E3 seed {args.seed} already complete; skipping.")
        return 0
    if (folder / "metrics.json").exists():
        raise FileExistsError(f"{folder} already contains metrics; use --resume or a fresh --out")
    folder.mkdir(parents=True, exist_ok=True)
    log = setup_logger(f"a2.e3.{args.seed}", folder / "run.log")
    started = time.perf_counter()
    log.info("E3 seed=%d device=%s config=%s", args.seed, device, args.config)
    result = run(cfg, args.seed, device, log)
    dump_runtime_config(cfg, folder / "config.yaml")
    save_pnl(folder / "pnl.npz", result["terminal_loss"], **result.get("extra_pnl", {}))
    for name, state in result.get("artifacts", {}).items():
        torch.save(state, folder / name)
    record = write_metrics(
        folder / "metrics.json",
        exp="e3",
        seed=args.seed,
        cfg=cfg,
        metrics=result["metrics"],
        checks=result["checks"],
        runtime_sec=time.perf_counter() - started,
        device=device,
        n_paths=result["n_paths"],
    )
    log.info(
        "E3 completed runtime=%s passed=%s",
        human_secs(time.perf_counter() - started),
        record["passed"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
