"""训练入口：一条命令跑完一个 (exp, seed) 的全部落盘（spec §4）。

用法（仓库根目录）：
    python -m src.replication.train --config configs/exp/debug.yaml
    python -m src.replication.train --config configs/exp/debug.yaml --seed 2 --out results.tmp
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from src.common.config import dump_runtime_config, load_config, resolve_seeds
from src.common.logging import human_secs, setup_logger
from src.common.metrics import save_pnl, write_metrics
from src.common.seed import resolve_device, seed_everything
from src.common.types import RunPaths
from src.replication import e1, e2, smoke

Runner = Callable[[dict[str, Any], int, torch.device, logging.Logger], dict[str, Any]]

# kind -> runner；新增环境（heston / bs）在这里注册，config 里用 kind 选择。
RUNNERS: dict[str, Runner] = {"smoke": smoke.run, "e1": e1.run, "e1d1": e1.run, "e2": e2.run}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deep Hedging experiment runner (SDSC4107).")
    parser.add_argument("--config", required=True, help="YAML 配置，如 configs/exp/debug.yaml")
    parser.add_argument("--seed", type=int, default=None, help="覆盖 config 里的 seeds")
    parser.add_argument("--out", default=None, help="覆盖 out_root（默认 results/）")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.out is not None:
        cfg["out_root"] = str(args.out)
    seeds = resolve_seeds(cfg, args.seed)
    if args.seed is not None:
        cfg["seeds"] = seeds
    device = resolve_device(cfg["device"])
    kind = str(cfg.get("kind", "smoke"))
    runner = RUNNERS.get(kind)
    if runner is None:
        raise SystemExit(f"未知 kind={kind!r}；已注册：{sorted(RUNNERS)}")

    for seed in seeds:
        started = time.perf_counter()
        paths = RunPaths(Path(str(cfg["out_root"])), str(cfg["exp"]), seed).mkdir()
        log = setup_logger(f"dh.{cfg['exp']}.{seed}", paths.log)
        dump_runtime_config(cfg, paths.config)
        seed_everything(seed)
        log.info("exp=%s seed=%d kind=%s device=%s", cfg["exp"], seed, kind, device)
        log.info("config=%s -> %s", args.config, paths.run_dir)
        payload = runner(cfg, seed, device, log)
        runtime = time.perf_counter() - started
        record = write_metrics(
            paths.metrics,
            exp=str(cfg["exp"]),
            seed=seed,
            cfg=cfg,
            metrics=payload["metrics"],
            checks=payload["checks"],
            runtime_sec=runtime,
            device=device,
            n_paths=int(payload["n_paths"]),
        )
        save_pnl(paths.pnl, payload["terminal_loss"], **payload.get("extra_pnl", {}))
        for name, obj in payload.get("artifacts", {}).items():
            torch.save(obj, paths.run_dir / name)
        log.info(
            "done runtime=%s passed=%s -> %s",
            human_secs(runtime),
            record["passed"],
            paths.metrics,
        )
        print(
            f"[{cfg['exp']}/{seed}] passed={record['passed']} "
            f"runtime={human_secs(runtime)} -> {paths.metrics}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
