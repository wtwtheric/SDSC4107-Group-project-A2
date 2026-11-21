"""Run an E4 epsilon grid; each epsilon/seed gets its own result directory."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from src.common.config import dump_runtime_config, load_config
from src.common.logging import human_secs, setup_logger
from src.common.metrics import save_pnl, write_metrics
from src.common.seed import resolve_device
from src.replication.e4 import run_one


def eps_tag(value: float) -> str:
    return f"{value:.10f}".rstrip("0").rstrip(".")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="e4bs.yaml or e4h.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    base_cfg = load_config(args.config)
    if args.device:
        base_cfg["device"] = args.device
    device = resolve_device(base_cfg["device"])
    eps_grid = [float(x) for x in base_cfg.get("eps_grid", [])]
    if len(eps_grid) < 2 or any(x <= 0.0 for x in eps_grid):
        raise ValueError("E4 config must define at least two positive eps_grid values")
    exp = str(base_cfg["exp"])
    summaries = []
    for eps in eps_grid:
        folder = Path(args.out) / exp / eps_tag(eps) / str(args.seed)
        metrics_path = folder / "metrics.json"
        if args.resume and metrics_path.exists():
            print(f"{exp} eps={eps:g} seed={args.seed} already complete; skipping.")
            summaries.append(json.loads(metrics_path.read_text(encoding="utf-8")))
            continue
        if metrics_path.exists():
            raise FileExistsError(f"{metrics_path} already exists; use --resume or fresh --out")
        folder.mkdir(parents=True, exist_ok=True)
        cfg = dict(base_cfg)
        cfg["eps"] = eps
        cfg.pop("eps_grid", None)
        log = setup_logger(f"a2.{exp}.{eps_tag(eps)}.{args.seed}", folder / "run.log")
        started = time.perf_counter()
        log.info("%s seed=%d eps=%.10g device=%s", exp, args.seed, eps, device)
        result = run_one(cfg, args.seed, device, log, eps)
        dump_runtime_config(cfg, folder / "config.yaml")
        save_pnl(folder / "pnl.npz", result["terminal_loss"])
        for name, state in result.get("artifacts", {}).items():
            torch.save(state, folder / name)
        record = write_metrics(
            metrics_path, exp=exp, seed=args.seed, cfg=cfg,
            metrics=result["metrics"], checks=result["checks"],
            runtime_sec=time.perf_counter() - started, device=device, n_paths=result["n_paths"],
        )
        summaries.append(record)
        log.info("%s eps=%g completed runtime=%s passed=%s", exp, eps,
                 human_secs(time.perf_counter() - started), record["passed"])
    summary_path = Path(args.out) / exp / f"seed_{args.seed}_epsilon_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"E4 summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
