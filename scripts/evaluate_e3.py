"""Re-price frozen E3 checkpoints on fresh calibration paths; reuse the original common OOS set."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.common.config import load_config
from src.common.metrics import git_provenance
from src.common.seed import resolve_device, seed_everything
from src.replication.e1 import market_from_cfg, policy_from_cfg, q_price
from src.replication.env_heston import HestonWorld
from src.replication.e3_evaluation import build_e3_summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_policy(
    checkpoint: Path,
    cfg: dict[str, Any],
    market: Any,
    device: torch.device,
    *,
    recurrent: bool,
) -> torch.nn.Module:
    policy = policy_from_cfg(
        cfg, market.n_features, market.d, device, recurrent=recurrent
    )
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    policy.load_state_dict(state)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


@torch.no_grad()
def _calibrate_policy(
    world: HestonWorld,
    policy: torch.nn.Module,
    *,
    n_paths: int,
    batch_size: int,
    seed: int,
    label: str,
) -> np.ndarray:
    """Generate fresh calibration losses for a frozen policy."""
    seed_everything(seed)
    losses = np.empty(n_paths, dtype=np.float64)
    cursor = 0
    while cursor < n_paths:
        size = min(batch_size, n_paths - cursor)
        paths = world.simulate(size)
        features = world.features(paths)
        positions = policy.forward_path(features)
        batch_loss = world.terminal_loss(positions, paths)
        losses[cursor : cursor + size] = batch_loss.detach().cpu().numpy()
        cursor += size
        if cursor == n_paths or cursor % (batch_size * 4) == 0:
            print(f"{label} calibration {cursor:,}/{n_paths:,}", flush=True)
    return losses


@torch.no_grad()
def evaluate(
    *,
    run_dir: Path,
    output_dir: Path,
    calibration_paths: int,
    batch_size: int,
    calibration_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run fresh independent calibration and score existing paired OOS loss arrays."""
    if calibration_paths < 1 or batch_size < 1:
        raise ValueError("calibration_paths and batch_size must both be positive")

    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.json"
    pnl_path = run_dir / "pnl.npz"
    recurrent_path = run_dir / "policy_recurrent.pt"
    simpler_path = run_dir / "policy_simpler.pt"
    for path in (config_path, metrics_path, pnl_path, recurrent_path, simpler_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required E3 source artifact not found: {path}")

    cfg = load_config(config_path)
    market = market_from_cfg(cfg)
    if market.d != 2 or market.eps <= 0.0:
        raise ValueError("E3 evaluation requires Heston d=2 and positive transaction costs")
    alpha = float(cfg["alpha"])
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    source_eval_seed = int((cfg.get("train", {}) or {}).get("eval_seed", 314159))
    if calibration_seed == source_eval_seed:
        raise ValueError("calibration_seed must differ from the original E3 OOS eval_seed")

    source_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with np.load(pnl_path, allow_pickle=False) as archive:
        required = ("terminal_loss", "terminal_loss_simpler")
        missing = [key for key in required if key not in archive]
        if missing:
            raise KeyError(f"E3 pnl.npz is missing required arrays: {missing}")
        oos_recurrent = np.asarray(archive["terminal_loss"], dtype=np.float64)
        oos_simpler = np.asarray(archive["terminal_loss_simpler"], dtype=np.float64)

    if oos_recurrent.ndim != 1 or oos_simpler.ndim != 1:
        raise ValueError("E3 OOS terminal-loss arrays must be one-dimensional")
    if oos_recurrent.shape != oos_simpler.shape:
        raise ValueError("Recurrent and simpler OOS arrays must have equal shape")
    if not np.isfinite(oos_recurrent).all() or not np.isfinite(oos_simpler).all():
        raise ValueError("OOS terminal-loss arrays contain non-finite values")
    if len(oos_recurrent) != int(source_metrics["n_paths"]):
        raise ValueError("OOS loss count does not match source metrics.json n_paths")

    world = HestonWorld(market, device=device)
    recurrent = _load_policy(recurrent_path, cfg, market, device, recurrent=True)
    simpler = _load_policy(simpler_path, cfg, market, device, recurrent=False)
    q = q_price(world)

    calibration_r = _calibrate_policy(
        world, recurrent, n_paths=calibration_paths, batch_size=batch_size,
        seed=calibration_seed, label="recurrent",
    )
    calibration_s = _calibrate_policy(
        world, simpler, n_paths=calibration_paths, batch_size=batch_size,
        seed=calibration_seed, label="simpler",
    )
    # The helper reseeds before each model, so both use identical fresh calibration paths.
    summary = build_e3_summary(
        calibration_r, calibration_s, oos_recurrent, oos_simpler, q=q, alpha=alpha
    )
    price_r = summary["recurrent"]["price"]
    price_s = summary["simpler"]["price"]
    summary["comparison"] = {
        "price_difference_simpler_minus_recurrent": float(price_s - price_r),
        "price_difference_pct_of_simpler": float((price_s - price_r) / price_s * 100.0),
        "mean_abs_error_reduction_pct": float(
            (summary["simpler"]["mean_abs_hedging_error"]
             - summary["recurrent"]["mean_abs_hedging_error"])
            / summary["simpler"]["mean_abs_hedging_error"] * 100.0
        ),
        "realized_cvar_recurrent_minus_simpler": float(
            summary["recurrent"]["realized_cvar"] - summary["simpler"]["realized_cvar"]
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "pnl.npz",
        calibration_loss_recurrent=calibration_r,
        calibration_loss_simpler=calibration_s,
        oos_terminal_loss_recurrent=oos_recurrent,
        oos_terminal_loss_simpler=oos_simpler,
        hedging_error_recurrent=price_r - oos_recurrent,
        hedging_error_simpler=price_s - oos_simpler,
    )

    payload = {
        "schema_version": "1.0",
        "exp": "e3_independent_evaluation",
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "device": str(device),
        "torch_version": torch.__version__,
        **git_provenance(),
        "source": {
            "exp": source_metrics.get("exp"),
            "seed": int(source_metrics.get("seed", 0)),
            "git_commit": source_metrics.get("git_commit"),
            "config_sha256": source_metrics.get("config_sha256"),
            "source_n_paths": len(oos_recurrent),
            "recurrent_checkpoint_sha256": _sha256(recurrent_path),
            "simpler_checkpoint_sha256": _sha256(simpler_path),
            "source_pnl_sha256": _sha256(pnl_path),
        },
        "calibration": {
            "seed": int(calibration_seed),
            "n_paths": int(calibration_paths),
            "batch_size": int(batch_size),
            "q_zero_cost_reference": float(q),
            "alpha": alpha,
            "price_formula": "q + CVaR_alpha(L_cal - q)",
            "same_paths_for_both_policies": True,
        },
        "oos": {
            "source": str(pnl_path),
            "n_paths": len(oos_recurrent),
            "reused_existing_common_oos_paths": True,
            "hedging_error_formula": "H = p_cal - L_oos",
            "realized_cvar_formula": "-CVaR_alpha(L_oos - p_cal)",
        },
        "mean_loss_note": (
            "The paper's Figure 5 table does not define Mean Loss in its caption. "
            "This workflow reports mean absolute hedging error E[|p_cal - L_oos|] "
            "as an explicitly labelled proxy, not as an exact reproduction of that column."
        ),
        "metrics": summary,
        "checks": {
            "finite_prices": bool(np.isfinite([price_r, price_s]).all()),
            "independent_calibration_and_oos_seed": bool(
                calibration_seed != source_eval_seed
            ),
            "common_oos_path_count": len(oos_recurrent) == len(oos_simpler),
            "common_calibration_paths": len(calibration_r) == len(calibration_s),
            "finite_oos_metrics": bool(
                np.isfinite([
                    summary["recurrent"]["mean_abs_hedging_error"],
                    summary["recurrent"]["realized_cvar"],
                    summary["simpler"]["mean_abs_hedging_error"],
                    summary["simpler"]["realized_cvar"],
                ]).all()
            ),
        },
    }
    payload["passed"] = all(payload["checks"].values())
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved independent E3 evaluation to {output_dir}", flush=True)
    for label in ("recurrent", "simpler"):
        row = summary[label]
        print(
            f"{label}: price={row['price']:.6f}, "
            f"mean_abs_H={row['mean_abs_hedging_error']:.6f}, "
            f"realized_CVaR={row['realized_cvar']:.6f}",
            flush=True,
        )
    print(f"checks passed: {payload['passed']}", flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently calibrate frozen E3 policies and evaluate the saved common OOS set."
    )
    parser.add_argument("--run-dir", type=Path, default=Path("results/e3/0"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--calibration-paths", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=25_000)
    parser.add_argument("--calibration-seed", type=int, default=314160)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir or args.run_dir / "independent_evaluation"
    evaluate(
        run_dir=args.run_dir,
        output_dir=output_dir,
        calibration_paths=args.calibration_paths,
        batch_size=args.batch_size,
        calibration_seed=args.calibration_seed,
        device=resolve_device(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
