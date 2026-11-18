"""Independent post-training BatchNorm calibration ablation for E2cs seed 0.

Calibration paths and OOS paths use distinct RNG seeds. The script preserves
original checkpoints and writes recalibrated copies to a separate output tree.
Only BN running_mean/running_var buffers are recalibrated; no gradients or
optimizer steps are performed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.common.config import load_config
from src.common.metrics import cvar
from src.common.seed import seed_everything
from src.replication.bn_recalibration import recalibrate_batchnorm_from_features
from src.replication.e1 import market_from_cfg, policy_from_cfg
from src.replication.e2cs import CallSpreadWorld


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_policy(folder: Path, device: torch.device) -> tuple[dict[str, Any], torch.nn.Module]:
    cfg = load_config(str(folder / "config.yaml"))
    if cfg["policy"].get("recurrent", True) or cfg["policy"].get("norm") != "batch":
        raise ValueError(f"{folder}: expected simpler BatchNorm policy")
    market = market_from_cfg(cfg)
    policy = policy_from_cfg(
        cfg, market.n_features, market.d, device, recurrent=False, n_features_override=2
    )
    policy.load_state_dict(
        torch.load(folder / "policy.pt", map_location=device, weights_only=True), strict=True
    )
    policy.eval()
    return cfg, policy


def _market_world(cfg: dict[str, Any]) -> CallSpreadWorld:
    market = market_from_cfg(cfg)
    return CallSpreadWorld(
        market,
        float(cfg["payoff"]["strike1"]),
        float(cfg["payoff"]["strike2"]),
        device="cpu",
    )


def _simulate_calibration_features(
    world: CallSpreadWorld, n_paths: int, batch_size: int, seed: int
) -> torch.Tensor:
    if n_paths < 2 or batch_size < 1:
        raise ValueError("calibration paths must be >=2 and batch size >=1")
    seed_everything(seed)
    features: list[torch.Tensor] = []
    for start in range(0, n_paths, batch_size):
        paths = world.simulate(min(batch_size, n_paths - start))
        features.append(world.features(paths).detach().cpu())
    return torch.cat(features, dim=0)


def _loss_summary(losses: np.ndarray, alpha: float) -> dict[str, float]:
    tensor = torch.as_tensor(losses, dtype=torch.float64)
    return {
        "mean_loss": float(tensor.mean()),
        "std_loss": float(tensor.std(unbiased=False)),
        "var_alpha": float(torch.quantile(tensor, alpha)),
        "cvar95": float(cvar(tensor, 0.95)),
        "cvar99": float(cvar(tensor, 0.99)),
        "loss_min": float(tensor.min()),
        "loss_max": float(tensor.max()),
    }


@torch.no_grad()
def _evaluate_pair(
    world: CallSpreadWorld,
    original: torch.nn.Module,
    calibrated: torch.nn.Module,
    *,
    n_paths: int,
    batch_size: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    seed_everything(seed)
    original.eval()
    calibrated.eval()
    base_losses: list[np.ndarray] = []
    cal_losses: list[np.ndarray] = []
    base_positions: list[np.ndarray] = []
    cal_positions: list[np.ndarray] = []
    for start in range(0, n_paths, batch_size):
        size = min(batch_size, n_paths - start)
        paths = world.simulate(size)
        features = world.features(paths)
        base_pos = original.forward_path(features)
        cal_pos = calibrated.forward_path(features)
        base_loss = world.terminal_loss(base_pos, paths)
        cal_loss = world.terminal_loss(cal_pos, paths)
        base_losses.append(base_loss.cpu().numpy())
        cal_losses.append(cal_loss.cpu().numpy())
        base_positions.append(base_pos[:, 15, :].cpu().numpy())
        cal_positions.append(cal_pos[:, 15, :].cpu().numpy())

    base_loss = np.concatenate(base_losses)
    cal_loss = np.concatenate(cal_losses)
    base_pos = np.concatenate(base_positions)
    cal_pos = np.concatenate(cal_positions)
    q = float(np.quantile(base_loss, alpha))
    base_tail = base_loss >= q
    cal_tail = cal_loss >= float(np.quantile(cal_loss, alpha))
    overlap = float(np.logical_and(base_tail, cal_tail).sum() / max(1, base_tail.sum()))
    return {
        "n_paths": n_paths,
        "seed": seed,
        "original": _loss_summary(base_loss, alpha),
        "recalibrated": _loss_summary(cal_loss, alpha),
        "paired_loss_difference_recalibrated_minus_original": {
            "mean": float(np.mean(cal_loss - base_loss)),
            "std": float(np.std(cal_loss - base_loss)),
            "p05": float(np.quantile(cal_loss - base_loss, 0.05)),
            "p50": float(np.quantile(cal_loss - base_loss, 0.50)),
            "p95": float(np.quantile(cal_loss - base_loss, 0.95)),
        },
        "worst_alpha_tail_overlap_fraction": overlap,
        "k15_positions": {
            "original_mean_abs": np.mean(np.abs(base_pos), axis=0).tolist(),
            "recalibrated_mean_abs": np.mean(np.abs(cal_pos), axis=0).tolist(),
            "mean_abs_change": np.mean(np.abs(cal_pos - base_pos), axis=0).tolist(),
            "stock_correlation": float(np.corrcoef(base_pos[:, 0], cal_pos[:, 0])[0, 1]),
            "varswap_correlation": float(np.corrcoef(base_pos[:, 1], cal_pos[:, 1])[0, 1]),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.checkpoint_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    results: dict[str, Any] = {
        "experiment": "E2cs independent post-training BatchNorm recalibration",
        "calibration_seed": args.calibration_seed,
        "oos_seed": args.oos_seed,
        "calibration_paths": args.calibration_paths,
        "oos_paths": args.oos_paths,
        "calibration_batch_size": args.calibration_batch_size,
        "oos_batch_size": args.oos_batch_size,
        "step": 15,
        "device": str(device),
        "policies": {},
        "limitations": [
            "This is a post-training BN recalibration ablation, not a retrained policy.",
            "Calibration and OOS paths use separate seeds and are not the original 1M-path evaluation sample.",
            "Only BN running_mean and running_var buffers are changed; weights and BN affine parameters remain frozen.",
            "A positive result does not prove the original training was faulty; a negative result does not prove BN irrelevant.",
        ],
    }

    for alpha in ("0.95", "0.99"):
        folder = root / alpha / str(args.seed)
        cfg, original = _load_policy(folder, device)
        world = _market_world(cfg)
        calibration_features = _simulate_calibration_features(
            world,
            args.calibration_paths,
            args.calibration_batch_size,
            args.calibration_seed,
        )
        calibrated = copy.deepcopy(original)
        calibration_report = recalibrate_batchnorm_from_features(
            calibrated,
            calibration_features,
            batch_size=args.calibration_batch_size,
            device=device,
        )
        # Ensure trainable/affine parameters are byte-identical.
        for name, value in original.state_dict().items():
            if name.endswith(("running_mean", "running_var")):
                continue
            torch.testing.assert_close(value, calibrated.state_dict()[name], rtol=0, atol=0)

        policy_out = output / "checkpoints" / alpha / str(args.seed)
        policy_out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / "config.yaml", policy_out / "config.yaml")
        torch.save(calibrated.state_dict(), policy_out / "policy.pt")
        results["policies"][alpha] = {
            "source_checkpoint": str(folder / "policy.pt"),
            "source_sha256": _sha256(folder / "policy.pt"),
            "recalibrated_checkpoint": str(policy_out / "policy.pt"),
            "recalibrated_sha256": _sha256(policy_out / "policy.pt"),
            "calibration": calibration_report,
        }
        del calibration_features

    # A separate OOS seed and common paths for original/recalibrated policies.
    for alpha in ("0.95", "0.99"):
        cfg, original = _load_policy(root / alpha / str(args.seed), device)
        calibrated_folder = output / "checkpoints" / alpha / str(args.seed)
        calibrated = copy.deepcopy(original)
        calibrated.load_state_dict(
            torch.load(calibrated_folder / "policy.pt", map_location=device, weights_only=True),
            strict=True,
        )
        world = _market_world(cfg)
        results["policies"][alpha]["oos"] = _evaluate_pair(
            world,
            original,
            calibrated,
            n_paths=args.oos_paths,
            batch_size=args.oos_batch_size,
            seed=args.oos_seed,
            alpha=float(alpha),
        )

    report_path = output / "bn_recalibration_report.json"
    report_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", default="backups/e2cs_simple_seed0/e2cs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-paths", type=int, default=200_000)
    parser.add_argument("--calibration-batch-size", type=int, default=25_000)
    parser.add_argument("--calibration-seed", type=int, default=928_372)
    parser.add_argument("--oos-paths", type=int, default=1_000_000)
    parser.add_argument("--oos-batch-size", type=int, default=25_000)
    parser.add_argument("--oos-seed", type=int, default=314_160)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/e2cs_bn_calibration_seed0")
    args = parser.parse_args()
    result = run(args)
    for alpha, item in result["policies"].items():
        before = item["oos"]["original"]
        after = item["oos"]["recalibrated"]
        print(
            f"alpha={alpha} CVaR{alpha} {before['cvar99']:.6f} -> "
            f"{after['cvar99']:.6f}; mean loss {before['mean_loss']:.6f} -> "
            f"{after['mean_loss']:.6f}"
        )
    print(f"Saved {args.output}/bn_recalibration_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
