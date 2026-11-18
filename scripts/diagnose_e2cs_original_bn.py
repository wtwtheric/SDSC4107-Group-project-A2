"""Original E2cs NPZ audit of frozen vs sample-moment BatchNorm, without training.

Requires corrected simpler seed-0 .pt checkpoints and BOTH original 200K visual
NPZs. The batch-moment counterfactual is deliberately diagnostic: it uses OOS
sample statistics and cannot be used as a new tradable strategy or to claim
improved OOS CVaR without a separate, properly calibrated evaluation.

Example:
python -m scripts.diagnose_e2cs_original_bn \
  --npz-95 /path/e2cs_095_pnl.npz --npz-99 /path/e2cs_099_pnl.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scripts.diagnose_e2cs_support_bn import _load_policies
from torch.nn import functional as F


def _load_data(path: Path) -> dict[str, np.ndarray]:
    fields = (
        "visual_spot",
        "visual_variance",
        "visual_stock_delta",
        "visual_varswap_delta",
        "model_spot_grid",
        "model_variance_grid",
        "model_stock_delta_grid",
        "model_varswap_delta_grid",
    )
    with np.load(path, allow_pickle=False) as z:
        missing = set(fields) - set(z.files)
        if missing:
            raise ValueError(f"{path} missing NPZ fields: {sorted(missing)}")
        return {key: np.asarray(z[key]) for key in fields}


@torch.no_grad()
def frozen_and_batch_moments(
    step_net: torch.nn.Module, state: torch.Tensor, out_scale: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Use actual frozen weights; never change checkpoint buffers or parameters."""
    step_net.eval()
    original_buffers = {name: value.detach().clone() for name, value in step_net.named_buffers()}
    z1 = step_net.fc1(state)
    y1_frozen = torch.relu(step_net.bn1(z1))
    z2 = step_net.fc2(y1_frozen)
    y2_frozen = torch.relu(step_net.bn2(z2))
    frozen = step_net.out(y2_frozen) * out_scale

    layers: dict[str, Any] = {}
    for name, pre in (("bn1", z1), ("bn2", z2)):
        bn = getattr(step_net, name)
        mu = pre.mean(0)
        var = pre.var(0, unbiased=False)
        shifts = (mu - bn.running_mean).abs() / torch.sqrt(bn.running_var + bn.eps)
        ratios = var / (bn.running_var + bn.eps)
        layers[name] = {
            "num_batches_tracked": int(bn.num_batches_tracked),
            "standardized_mean_shift_median": float(shifts.median()),
            "standardized_mean_shift_max": float(shifts.max()),
            "variance_ratio_median": float(ratios.median()),
            "variance_ratio_min": float(ratios.min()),
            "variance_ratio_max": float(ratios.max()),
            "running_var_min": float(bn.running_var.min()),
        }

    # BN1 and BN2 both recomputed from the SAME independent, complete OOS
    # visual-state sample, without mutating moving averages.
    y1 = torch.relu(
        F.batch_norm(
            z1,
            None,
            None,
            step_net.bn1.weight,
            step_net.bn1.bias,
            training=True,
            eps=step_net.bn1.eps,
        )
    )
    z2_new = step_net.fc2(y1)
    y2 = torch.relu(
        F.batch_norm(
            z2_new,
            None,
            None,
            step_net.bn2.weight,
            step_net.bn2.bias,
            training=True,
            eps=step_net.bn2.eps,
        )
    )
    alternative = step_net.out(y2) * out_scale
    for name, value in step_net.named_buffers():
        torch.testing.assert_close(value, original_buffers[name])
    return (
        frozen.cpu().numpy(),
        alternative.cpu().numpy(),
        layers,
    )


def _grid_error(
    s: np.ndarray,
    v: np.ndarray,
    values: np.ndarray,
    sg: np.ndarray,
    vg: np.ndarray,
    model_grid: np.ndarray,
) -> dict[str, Any]:
    inside = (s >= sg[0]) & (s <= sg[-1]) & (v >= vg[0]) & (v <= vg[-1])
    if not inside.any():
        raise ValueError("No market states within model hedge grid")
    interp = RegularGridInterpolator(
        (sg, vg),
        model_grid,
        bounds_error=True,
    )(np.column_stack((s[inside], v[inside])))
    error = values[inside] - interp
    return {
        "n_inside": int(inside.sum()),
        "fraction_inside": float(inside.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mean_signed_error": float(error.mean()),
    }


def analyze(
    *,
    npz95: Path,
    npz99: Path,
    checkpoint_root: Path,
    output: Path,
    seed: int = 0,
    step: int = 15,
) -> dict[str, Any]:
    a = _load_data(npz95)
    b = _load_data(npz99)
    if any(
        not np.array_equal(a[key], b[key])
        for key in (
            "visual_spot",
            "visual_variance",
            "model_spot_grid",
            "model_variance_grid",
            "model_stock_delta_grid",
            "model_varswap_delta_grid",
        )
    ):
        raise ValueError("The two policies do not share identical visual paths/model grids")
    s = np.asarray(a["visual_spot"], dtype=np.float64)
    v = np.asarray(a["visual_variance"], dtype=np.float64)
    if len(s) < 2 or not np.isfinite(s).all() or not np.isfinite(v).all():
        raise ValueError("Invalid visual market states")
    state = torch.from_numpy(np.column_stack((s, v))).to(torch.float64)
    policies = _load_policies(checkpoint_root, seed, torch.device("cpu"))
    results: dict[str, Any] = {}
    for alpha, saved in (("0.95", a), ("0.99", b)):
        policy = policies[alpha]
        if policy.recurrent or policy.n_features != 2:
            raise ValueError("Expected simpler two-state policy")
        original, alternative, bn = frozen_and_batch_moments(
            policy.steps[step], state, policy.out_scale
        )
        stored = np.column_stack(
            (saved["visual_stock_delta"], saved["visual_varswap_delta"])
        ).astype(np.float64)
        consistency = np.abs(original - stored)
        original_errors = {
            "stock": _grid_error(
                s,
                v,
                original[:, 0],
                a["model_spot_grid"],
                a["model_variance_grid"],
                a["model_stock_delta_grid"],
            ),
            "variance_swap": _grid_error(
                s,
                v,
                original[:, 1],
                a["model_spot_grid"],
                a["model_variance_grid"],
                a["model_varswap_delta_grid"],
            ),
        }
        alternative_errors = {
            "stock": _grid_error(
                s,
                v,
                alternative[:, 0],
                a["model_spot_grid"],
                a["model_variance_grid"],
                a["model_stock_delta_grid"],
            ),
            "variance_swap": _grid_error(
                s,
                v,
                alternative[:, 1],
                a["model_spot_grid"],
                a["model_variance_grid"],
                a["model_varswap_delta_grid"],
            ),
        }
        results[alpha] = {
            "checkpoint_matches_archived_visual_positions": bool(np.max(consistency) < 1e-4),
            "max_abs_checkpoint_vs_archived_stock": float(np.max(consistency[:, 0])),
            "max_abs_checkpoint_vs_archived_varswap": float(np.max(consistency[:, 1])),
            "bn_moment_mismatch": bn,
            "mean_abs_change_stock_with_sample_bn": float(
                np.mean(np.abs(alternative[:, 0] - original[:, 0]))
            ),
            "mean_abs_change_varswap_with_sample_bn": float(
                np.mean(np.abs(alternative[:, 1] - original[:, 1]))
            ),
            "frozen_grid_errors": original_errors,
            "sample_bn_grid_errors_diagnostic_only": alternative_errors,
        }
    report = {
        "sample": {"n_visual_paths": len(s), "step": step, "seed": seed},
        "input_scale": {
            "spot_mean": float(np.mean(s)),
            "spot_std": float(np.std(s)),
            "variance_mean": float(np.mean(v)),
            "variance_std": float(np.std(v)),
            "variance_exact_zero_fraction": float(np.mean(v == 0)),
        },
        "policies": results,
        "limitations": [
            "Changing input normalization requires new training; cannot transform existing checkpoints.",
            "This diagnostic recalibrates BN using evaluation-state inputs, not a train-time dataset.",
            "Counterfactual moments are not a valid deployed strategy or OOS CVaR estimate.",
            "The provided NPZ contains only k=15 states and terminal losses, not complete path trajectories.",
            "Model hedge comparisons apply only within the original Figure 8/9 grid.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-95", type=Path, required=True)
    parser.add_argument("--npz-99", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("backups/e2cs_simple_seed0/e2cs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/e2cs/e2cs_original_bn_counterfactual.json"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=int, default=15)
    args = parser.parse_args()
    result = analyze(
        npz95=args.npz_95,
        npz99=args.npz_99,
        checkpoint_root=args.checkpoint_root,
        output=args.output,
        seed=args.seed,
        step=args.step,
    )
    for alpha, record in result["policies"].items():
        print(
            f"{alpha}: saved checkpoint match={record['checkpoint_matches_archived_visual_positions']} "
            f"BN2 shift={record['bn_moment_mismatch']['bn2']['standardized_mean_shift_median']:.4f} "
            f"stock RMSE frozen={record['frozen_grid_errors']['stock']['rmse']:.5f} "
            f"sample moments={record['sample_bn_grid_errors_diagnostic_only']['stock']['rmse']:.5f}"
        )
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
