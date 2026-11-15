"""Compare old ceil-CVaR and new fractional-CVaR E2 alpha=0.99 policies.

Both frozen checkpoints are evaluated on the same fresh 1M QE-M Heston paths.
At 1M paths and alpha=0.99 the tail mass is exactly 10,000, so the evaluation
functional itself is identical for ceil and fractional estimators. Any
difference therefore comes from the training estimator, not test-set rounding.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
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


def _load_policy(
    checkpoint: Path,
    cfg: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    market = market_from_cfg(cfg)
    policy = policy_from_cfg(cfg, market.n_features, market.d, device, recurrent=True)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    policy.load_state_dict(state)
    policy.eval()
    return policy


def _metrics(loss: np.ndarray, q: float) -> dict[str, float]:
    tensor = torch.from_numpy(loss)
    excess = tensor - q
    return {
        "price_alpha99": float(cvar(tensor, 0.99)),
        "cvar50_excess": float(cvar(excess, 0.50)),
        "cvar99_excess": float(cvar(excess, 0.99)),
        "mean_raw_loss": float(tensor.mean()),
        "std_raw_loss": float(tensor.std(unbiased=False)),
        "mean_abs_hedging_error": float((q - tensor).abs().mean()),
    }


@torch.no_grad()
def compare(
    *,
    old_run_dir: Path,
    new_run_dir: Path,
    output_dir: Path,
    seed: int,
    n_paths: int,
    batch_size: int,
    evaluation_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if n_paths < 1 or batch_size < 1:
        raise ValueError("n_paths and batch_size must both be positive")

    old_cfg_path = old_run_dir / "config.yaml"
    new_cfg_path = new_run_dir / "config.yaml"
    old_checkpoint = old_run_dir / "policy_alpha99.pt"
    new_checkpoint = new_run_dir / "policy_alpha99.pt"
    for path in (old_cfg_path, new_cfg_path, old_checkpoint, new_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    old_cfg = load_config(old_cfg_path)
    new_cfg = load_config(new_cfg_path)
    old_market = market_from_cfg(old_cfg)
    new_market = market_from_cfg(new_cfg)
    if old_market != new_market:
        raise ValueError("old/new Heston market configs differ; controlled comparison is invalid")

    world = HestonWorld(old_market, device=device)
    old_policy = _load_policy(old_checkpoint, old_cfg, device)
    new_policy = _load_policy(new_checkpoint, new_cfg, device)
    q = q_price(world)

    losses_old = np.empty(n_paths, dtype=np.float64)
    losses_new = np.empty(n_paths, dtype=np.float64)

    seed_everything(evaluation_seed)
    cursor = 0
    while cursor < n_paths:
        current = min(batch_size, n_paths - cursor)
        paths = world.simulate(current)
        features = world.features(paths)

        positions_old = old_policy.forward_path(features)
        positions_new = new_policy.forward_path(features)
        loss_old = world.terminal_loss(positions_old, paths)
        loss_new = world.terminal_loss(positions_new, paths)

        stop = cursor + current
        losses_old[cursor:stop] = loss_old.detach().cpu().numpy()
        losses_new[cursor:stop] = loss_new.detach().cpu().numpy()
        cursor = stop

        if cursor == n_paths or cursor % (batch_size * 4) == 0:
            print(f"evaluated {cursor:,}/{n_paths:,} shared paths", flush=True)

        del paths, features, positions_old, positions_new, loss_old, loss_new

    old_metrics = _metrics(losses_old, q)
    new_metrics = _metrics(losses_new, q)
    delta_metrics = {key: new_metrics[key] - old_metrics[key] for key in old_metrics}

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "exp": "e2_fractional_vs_ceil_seed0",
        "seed": int(seed),
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "evaluation_paths": int(n_paths),
        "batch_size": int(batch_size),
        "evaluation_seed": int(evaluation_seed),
        "device": str(device),
        "torch_version": torch.__version__,
        "simulator": "Andersen QE-M via pfhedge.generate_heston",
        "evaluation_cvar_estimator": "fractional_boundary",
        "evaluation_note": (
            "At n=1,000,000 and alpha=0.99 the tail mass is exactly 10,000, "
            "so ceil and fractional evaluation coincide."
        ),
        "q": q,
        "old_training_estimator": "ceil_boundary",
        "new_training_estimator": "fractional_boundary",
        "old_config_sha256": digest_config(old_cfg),
        "new_config_sha256": digest_config(new_cfg),
        "old_checkpoint_sha256": _sha256(old_checkpoint),
        "new_checkpoint_sha256": _sha256(new_checkpoint),
        **git_provenance(),
        "metrics_old_ceil": old_metrics,
        "metrics_new_fractional": new_metrics,
        "delta_new_minus_old": delta_metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "pnl.npz",
        terminal_loss_old_ceil=losses_old,
        terminal_loss_new_fractional=losses_new,
    )

    print(
        "old/new p99: "
        f"{old_metrics['price_alpha99']:.6f} -> {new_metrics['price_alpha99']:.6f} "
        f"(delta {delta_metrics['price_alpha99']:+.6f})",
        flush=True,
    )
    print(
        "old/new CVaR99(L-q): "
        f"{old_metrics['cvar99_excess']:.6f} -> {new_metrics['cvar99_excess']:.6f} "
        f"(delta {delta_metrics['cvar99_excess']:+.6f})",
        flush=True,
    )
    print(f"saved comparison to {output_dir}", flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="1M-path shared-scenario comparison: old ceil vs new fractional E2 alpha99."
    )
    parser.add_argument("--old-run-dir", type=Path, default=Path("backups/e2/0"))
    parser.add_argument("--new-run-dir", type=Path, default=Path("results/e2_fractional/0"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/diagnostics/e2_fractional_vs_ceil_seed0"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-paths", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=25_000)
    parser.add_argument("--evaluation-seed", type=int, default=10_001)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    compare(
        old_run_dir=args.old_run_dir,
        new_run_dir=args.new_run_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        n_paths=args.n_paths,
        batch_size=args.batch_size,
        evaluation_seed=args.evaluation_seed,
        device=resolve_device(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
