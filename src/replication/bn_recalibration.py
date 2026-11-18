"""Post-training BatchNorm recalibration from independent Heston calibration paths.

Only BatchNorm running_mean/running_var buffers are changed. Model weights,
BatchNorm affine parameters, input features, and the trading objective remain
frozen. Calibration features must be generated independently of OOS evaluation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


def _collect_module_outputs(
    module: nn.Module,
    inputs: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Apply a frozen module in chunks and return CPU float64 activations."""
    parts: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            chunk = inputs[start : start + batch_size].to(device=device, dtype=torch.float64)
            parts.append(module(chunk).detach().to(device="cpu", dtype=torch.float64))
    return torch.cat(parts, dim=0)


def _sample_moments(values: Tensor) -> tuple[Tensor, Tensor]:
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("BatchNorm calibration requires at least two 2D activation samples")
    mean = values.mean(dim=0)
    # PyTorch BatchNorm stores the unbiased variance estimate in running_var.
    variance = values.var(dim=0, unbiased=True)
    if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
        raise ValueError("Non-finite activation moments during BatchNorm calibration")
    return mean, variance


def _compare_moments(
    values: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    eps: float,
) -> dict[str, float]:
    mean, variance = _sample_moments(values)
    running_mean = running_mean.detach().cpu().to(torch.float64)
    running_var = running_var.detach().cpu().to(torch.float64)
    mean_shift = (mean - running_mean).abs() / torch.sqrt(running_var + eps)
    variance_ratio = variance / (running_var + eps)
    return {
        "standardized_mean_shift_median": float(mean_shift.median()),
        "standardized_mean_shift_max": float(mean_shift.max()),
        "variance_ratio_median": float(variance_ratio.median()),
        "variance_ratio_min": float(variance_ratio.min()),
        "variance_ratio_max": float(variance_ratio.max()),
    }


@torch.no_grad()
def recalibrate_batchnorm_from_features(
    policy: nn.Module,
    features: Tensor,
    *,
    batch_size: int = 25_000,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Re-estimate each date's BN moments from independent (paths, dates, 2) data.

    The calibration procedure is sequential within each StepMLP:
      1. Estimate BN1 moments from fc1(x_k) across all calibration paths.
      2. Install those BN1 moments, then run BN1 in eval mode.
      3. Estimate BN2 moments from fc2(ReLU(BN1(fc1(x_k)))).
      4. Install BN2 moments.

    The estimates are exact full-sample moments over the supplied calibration
    paths (not an EMA of mini-batches). The unbiased sample variance matches
    PyTorch's running_var convention. No optimizer or backward pass is used.
    """
    target_device = torch.device(device)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if features.ndim != 3 or features.shape[-1] != 2:
        raise ValueError("features must have shape (n_paths, n_steps, 2) for raw (S,V)")
    if features.shape[0] < 2:
        raise ValueError("Need at least two independent calibration paths")
    if not hasattr(policy, "steps") or len(policy.steps) != features.shape[1]:
        raise ValueError("Policy time-step count does not match calibration features")
    if bool(getattr(policy, "recurrent", True)):
        raise ValueError("Expected the E2cs simpler (non-recurrent) policy")
    if str(getattr(policy, "norm_kind", "")) != "batch":
        raise ValueError("Expected a BatchNorm policy")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError("Calibration features must be finite floating-point values")

    policy.to(device=target_device, dtype=torch.float64)
    policy.eval()
    features_cpu = features.detach().to(device="cpu", dtype=torch.float64).contiguous()

    state_before = {
        name: value.detach().cpu().clone() for name, value in policy.state_dict().items()
    }
    report: list[dict[str, Any]] = []

    for step_index, step_net in enumerate(policy.steps):
        x = features_cpu[:, step_index, :]
        z1 = _collect_module_outputs(
            step_net.fc1, x, batch_size=batch_size, device=target_device
        )
        bn1 = step_net.bn1
        bn2 = step_net.bn2
        if bn1.running_mean is None or bn1.running_var is None:
            raise ValueError(f"Step {step_index}: BN1 does not track running statistics")
        if bn2.running_mean is None or bn2.running_var is None:
            raise ValueError(f"Step {step_index}: BN2 does not track running statistics")

        bn1_before = _compare_moments(
            z1, bn1.running_mean, bn1.running_var, bn1.eps
        )
        bn1_mean, bn1_var = _sample_moments(z1)
        bn1.running_mean.copy_(bn1_mean.to(device=target_device))
        bn1.running_var.copy_(bn1_var.to(device=target_device))
        bn1_after = _compare_moments(z1, bn1.running_mean, bn1.running_var, bn1.eps)

        # BN1 now uses its newly calibrated moments; this defines the input
        # distribution that BN2 must normalize at inference time.
        z2_parts: list[Tensor] = []
        with torch.no_grad():
            for start in range(0, z1.shape[0], batch_size):
                z1_chunk = z1[start : start + batch_size].to(device=target_device)
                hidden1 = torch.relu(bn1(z1_chunk))
                z2_parts.append(
                    step_net.fc2(hidden1).detach().to(device="cpu", dtype=torch.float64)
                )
        z2 = torch.cat(z2_parts, dim=0)
        bn2_before = _compare_moments(
            z2, bn2.running_mean, bn2.running_var, bn2.eps
        )
        bn2_mean, bn2_var = _sample_moments(z2)
        bn2.running_mean.copy_(bn2_mean.to(device=target_device))
        bn2.running_var.copy_(bn2_var.to(device=target_device))
        bn2_after = _compare_moments(z2, bn2.running_mean, bn2.running_var, bn2.eps)

        report.append(
            {
                "step": step_index,
                "n_calibration_paths": int(features_cpu.shape[0]),
                "bn1_before_recalibration": bn1_before,
                "bn1_after_recalibration": bn1_after,
                "bn2_before_recalibration_after_bn1_update": bn2_before,
                "bn2_after_recalibration": bn2_after,
                "bn1_calibrated_mean_abs": float(bn1_mean.abs().mean()),
                "bn1_calibrated_var_median": float(bn1_var.median()),
                "bn2_calibrated_mean_abs": float(bn2_mean.abs().mean()),
                "bn2_calibrated_var_median": float(bn2_var.median()),
                "bn1_num_batches_tracked_preserved": int(bn1.num_batches_tracked),
                "bn2_num_batches_tracked_preserved": int(bn2.num_batches_tracked),
            }
        )

    allowed_changed = {
        f"steps.{index}.{layer}.{buffer}"
        for index in range(len(policy.steps))
        for layer in ("bn1", "bn2")
        for buffer in ("running_mean", "running_var")
    }
    state_after = policy.state_dict()
    changed: list[str] = []
    for name, value in state_after.items():
        before = state_before[name]
        if not torch.equal(value.detach().cpu(), before):
            changed.append(name)
    unexpected = sorted(set(changed) - allowed_changed)
    if unexpected:
        raise RuntimeError(
            "Calibration changed non-running-statistics state: " + ", ".join(unexpected)
        )

    return {
        "n_calibration_paths": int(features_cpu.shape[0]),
        "n_steps": int(features_cpu.shape[1]),
        "changed_buffers": sorted(changed),
        "only_bn_running_moments_changed": set(changed).issubset(allowed_changed),
        "step_diagnostics": report,
    }
