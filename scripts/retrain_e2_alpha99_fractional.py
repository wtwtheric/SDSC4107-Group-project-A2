"""Retrain only the E2 alpha=0.99 policy with fractional empirical CVaR.

This is a controlled diagnostic. It keeps the E2 Heston QE-M simulator, seed,
network, optimizer, batch size and iteration budget fixed and changes only the
CVaR estimator in src.common.metrics.cvar.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from src.common.config import digest_config, dump_runtime_config, load_config
from src.common.logging import human_secs, setup_logger
from src.common.metrics import cvar, git_provenance, save_pnl
from src.common.seed import resolve_device, seed_everything
from src.replication.e1 import market_from_cfg, q_price, train_policy
from src.replication.e2 import config_with_alpha, evaluate_on_paths
from src.replication.env_heston import HestonWorld


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    *,
    config_path: Path,
    seed: int,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    market = market_from_cfg(cfg)
    if market.d != 2 or market.eps != 0.0:
        raise ValueError("E2 fractional-CVaR diagnostic requires Heston d=2 and zero costs")

    alpha = float(cfg.get("alpha", 0.99))
    if abs(alpha - 0.99) > 1e-12:
        raise ValueError(f"expected E2 alpha=0.99, got {alpha}")

    cfg99 = config_with_alpha(cfg, 0.99)
    cfg99["seeds"] = [int(seed)]
    train_cfg = cfg99.get("train", {}) or {}
    eval_paths = int(train_cfg.get("eval_paths", 100_000))
    eval_offset = int(train_cfg.get("eval_seed_offset", 10_000))

    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger(f"dh.e2.fractional.{seed}", output_dir / "run.log")
    dump_runtime_config(cfg99, output_dir / "config.yaml")

    world = HestonWorld(market, device=device)
    log.info("controlled E2 retrain: alpha=0.99, CVaR estimator=fractional_boundary")
    log.info(
        "simulator=Andersen QE-M seed=%d batch=%d n_iter=%d device=%s",
        seed,
        int(cfg99["batch"]),
        int(train_cfg.get("n_iter", 5000)),
        device,
    )

    started = time.perf_counter()
    seed_everything(seed)
    policy = train_policy(world, cfg99, device, log, recurrent=True)
    runtime_sec = time.perf_counter() - started

    checkpoint = output_dir / "policy_alpha99.pt"
    torch.save(policy.state_dict(), checkpoint)

    seed_everything(seed + eval_offset)
    paths = world.simulate(eval_paths)
    ev = evaluate_on_paths(world, policy, paths)
    loss = ev["loss"]
    q = q_price(world)
    price99 = float(cvar(loss, 0.99))
    excess = loss - q

    metrics = {
        "price_alpha99": price99,
        "q": q,
        "cvar50_alpha99": float(cvar(excess, 0.50)),
        "cvar99_alpha99": float(cvar(excess, 0.99)),
        "mean_raw_loss_alpha99": float(loss.mean()),
        "std_raw_loss_alpha99": float(loss.std(unbiased=False)),
        "mean_abs_hedging_error_alpha99": float((q - loss).abs().mean()),
    }
    checks = {
        "mean_loss_near_q": abs(metrics["mean_raw_loss_alpha99"] - q) < 0.02,
        "price_translation": abs(price99 - (q + metrics["cvar99_alpha99"])) < 1e-10,
    }

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "exp": "e2_fractional_diagnostic",
        "seed": int(seed),
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "runtime_sec": round(runtime_sec, 3),
        "device": str(device),
        "torch_version": torch.__version__,
        "simulator": "Andersen QE-M via pfhedge.generate_heston",
        "cvar_estimator": "fractional_boundary",
        "training_alpha": 0.99,
        "training_batch": int(cfg99["batch"]),
        "training_iterations": int(train_cfg.get("n_iter", 5000)),
        "evaluation_paths": eval_paths,
        "evaluation_seed": int(seed + eval_offset),
        "config_sha256": digest_config(cfg99),
        "checkpoint_sha256": _sha256(checkpoint),
        **git_provenance(),
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_pnl(output_dir / "pnl.npz", loss.detach().cpu().numpy())

    log.info(
        "quick eval p99=%.6f q=%.6f cvar99(L-q)=%.6f mean_loss=%.6f",
        price99,
        q,
        metrics["cvar99_alpha99"],
        metrics["mean_raw_loss_alpha99"],
    )
    log.info(
        "done runtime=%s checkpoint=%s passed=%s",
        human_secs(runtime_sec),
        checkpoint,
        payload["passed"],
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrain only E2 alpha=0.99 with fractional empirical CVaR."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/exp/e2.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/e2_fractional/0"))
    parser.add_argument("--device", default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(
        config_path=args.config,
        seed=args.seed,
        output_dir=args.output_dir,
        device=resolve_device(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
