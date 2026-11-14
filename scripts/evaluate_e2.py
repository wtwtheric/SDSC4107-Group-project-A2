"""Evaluate frozen E2 policies on a fresh, shared 1M-path Heston test set."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.common.config import digest_config, load_config
from src.common.metrics import cvar, git_provenance
from src.common.seed import resolve_device, seed_everything
from src.replication.e1 import market_from_cfg, policy_from_cfg, q_price
from src.replication.env_heston import HestonWorld


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tail_metrics(loss: np.ndarray, q: float, tag: str) -> dict[str, float]:
    excess_loss = loss - q
    out: dict[str, float] = {
        f"mean_raw_loss_{tag}": float(np.mean(loss)),
        f"std_raw_loss_{tag}": float(np.std(loss)),
        f"mean_abs_hedging_error_{tag}": float(np.mean(np.abs(q - loss))),
        f"min_excess_loss_{tag}": float(np.min(excess_loss)),
        f"max_excess_loss_{tag}": float(np.max(excess_loss)),
    }
    for alpha, label in ((0.95, "95"), (0.99, "99"), (0.995, "995"), (0.999, "999")):
        out[f"var{label}_{tag}"] = float(np.quantile(excess_loss, alpha))
        out[f"cvar{label}_{tag}"] = float(cvar(torch.from_numpy(excess_loss), alpha))
    return out


def _find_input(run_dir: Path, name: str, seed: int) -> Path:
    candidates = [run_dir / name, Path("backups") / "e2" / str(seed) / name]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Cannot find {name}. Looked in: " + ", ".join(str(p) for p in candidates)
    )


def _load_policy(
    checkpoint: Path,
    cfg: dict[str, Any],
    n_features: int,
    d: int,
    n_steps: int,
    device: torch.device,
) -> torch.nn.Module:
    policy = policy_from_cfg(cfg, n_features, d, device, recurrent=True)
    if policy.n_steps != n_steps:
        raise ValueError(f"checkpoint architecture n_steps={policy.n_steps}, config has {n_steps}")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    policy.load_state_dict(state)
    policy.eval()
    return policy


@torch.no_grad()
def evaluate(
    *,
    run_dir: Path,
    output_dir: Path,
    n_paths: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if n_paths < 1 or batch_size < 1:
        raise ValueError("n_paths and batch_size must both be positive")

    config_path = _find_input(run_dir, "config.yaml", seed)
    cfg = load_config(config_path)
    market = market_from_cfg(cfg)
    if market.d != 2 or market.eps != 0.0:
        raise ValueError("E2 evaluation requires Heston d=2 and zero transaction costs")

    checkpoint50 = _find_input(run_dir, "policy_alpha50.pt", seed)
    checkpoint99 = _find_input(run_dir, "policy_alpha99.pt", seed)
    world = HestonWorld(market, device=device)
    policy50 = _load_policy(
        checkpoint50, cfg, market.n_features, market.d, market.n_steps, device
    )
    policy99 = _load_policy(
        checkpoint99, cfg, market.n_features, market.d, market.n_steps, device
    )

    q = q_price(world)
    losses50 = np.empty(n_paths, dtype=np.float64)
    losses99 = np.empty(n_paths, dtype=np.float64)
    tags = ("alpha50", "alpha99")
    position_sums = {tag: np.zeros((market.n_steps, market.d)) for tag in tags}
    position_sums_sq = {tag: np.zeros((market.n_steps, market.d)) for tag in tags}
    position_mins = {tag: np.full((market.n_steps, market.d), np.inf) for tag in tags}
    position_maxs = {tag: np.full((market.n_steps, market.d), -np.inf) for tag in tags}

    eval_offset = int((cfg.get("train", {}) or {}).get("eval_seed_offset", 10_000))
    eval_seed = seed + eval_offset + 1  # separate from the original 100k evaluation
    seed_everything(eval_seed)

    cursor = 0
    while cursor < n_paths:
        current = min(batch_size, n_paths - cursor)
        paths = world.simulate(current)
        features = world.features(paths)
        positions50 = policy50.forward_path(features)
        positions99 = policy99.forward_path(features)
        loss50 = world.terminal_loss(positions50, paths)
        loss99 = world.terminal_loss(positions99, paths)

        stop = cursor + current
        losses50[cursor:stop] = loss50.detach().cpu().numpy()
        losses99[cursor:stop] = loss99.detach().cpu().numpy()

        for tag, positions in (("alpha50", positions50), ("alpha99", positions99)):
            values = positions.detach().cpu().numpy()
            position_sums[tag] += values.sum(axis=0)
            position_sums_sq[tag] += np.square(values).sum(axis=0)
            position_mins[tag] = np.minimum(position_mins[tag], values.min(axis=0))
            position_maxs[tag] = np.maximum(position_maxs[tag], values.max(axis=0))

        cursor = stop
        if cursor == n_paths or cursor % (batch_size * 4) == 0:
            print(f"evaluated {cursor:,}/{n_paths:,} paths", flush=True)
        del paths, features, positions50, positions99, loss50, loss99

    loss_tensor50 = torch.from_numpy(losses50)
    loss_tensor99 = torch.from_numpy(losses99)
    price50 = float(cvar(loss_tensor50, 0.50))
    price99 = float(cvar(loss_tensor99, 0.99))
    excess50 = losses50 - q
    excess99 = losses99 - q

    metrics: dict[str, float] = {
        "price": price99,
        "price_alpha50": price50,
        "price_alpha99": price99,
        "q": q,
        "cvar05_other": float(cvar(torch.from_numpy(excess50), 0.50)),
        "cvar99_other": float(cvar(torch.from_numpy(excess50), 0.99)),
        "cvar05_own": float(cvar(torch.from_numpy(excess99), 0.50)),
        "cvar99_own": float(cvar(torch.from_numpy(excess99), 0.99)),
    }
    metrics.update(_tail_metrics(losses50, q, "alpha50"))
    metrics.update(_tail_metrics(losses99, q, "alpha99"))

    checks = {
        "price99_band": 3.2 <= price99 <= 3.8,
        "price50_reference_band": 1.80 <= price50 <= 2.10,
        "cvar50_tradeoff": metrics["cvar05_own"] > metrics["cvar05_other"],
        "cvar99_tradeoff": metrics["cvar99_own"] < metrics["cvar99_other"],
        "price_translation_alpha50": abs(price50 - (q + metrics["cvar05_other"])) < 1e-10,
        "price_translation_alpha99": abs(price99 - (q + metrics["cvar99_own"])) < 1e-10,
    }

    position_arrays: dict[str, np.ndarray] = {}
    for tag in tags:
        mean = position_sums[tag] / n_paths
        variance = np.maximum(position_sums_sq[tag] / n_paths - np.square(mean), 0.0)
        position_arrays[f"position_mean_{tag}"] = mean
        position_arrays[f"position_std_{tag}"] = np.sqrt(variance)
        position_arrays[f"position_min_{tag}"] = position_mins[tag]
        position_arrays[f"position_max_{tag}"] = position_maxs[tag]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "pnl.npz",
        terminal_loss_alpha50=losses50,
        terminal_loss_alpha99=losses99,
        **position_arrays,
    )
    shutil.copy2(config_path, output_dir / "config.yaml")

    source_metrics_path = run_dir / "metrics.json"
    source_metrics: dict[str, Any] = {}
    if source_metrics_path.is_file():
        source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))

    payload = {
        "schema_version": "1.0",
        "exp": "e2",
        "seed": int(seed),
        "evaluation_paths": int(n_paths),
        "batch_size": int(batch_size),
        "evaluation_seed": int(eval_seed),
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "device": str(device),
        "torch_version": torch.__version__,
        **git_provenance(),
        "source_git_commit": source_metrics.get("git_commit"),
        "source_config_sha256": digest_config(cfg),
        "source_checkpoints": {
            "policy_alpha50_sha256": _sha256(checkpoint50),
            "policy_alpha99_sha256": _sha256(checkpoint99),
        },
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved evaluation to {output_dir}", flush=True)
    print(f"price50={price50:.6f}, price99={price99:.6f}, q={q:.6f}", flush=True)
    print(f"checks passed: {payload['passed']}", flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunked 1M-path evaluation for frozen E2 policies.")
    parser.add_argument("--run-dir", type=Path, default=Path("results/e2/0"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-paths", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seed = int(args.seed) if args.seed is not None else int(args.run_dir.name)
    output_dir = args.output_dir or args.run_dir / f"evaluation_{args.n_paths}"
    evaluate(
        run_dir=args.run_dir,
        output_dir=output_dir,
        n_paths=args.n_paths,
        batch_size=args.batch_size,
        seed=seed,
        device=resolve_device(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
