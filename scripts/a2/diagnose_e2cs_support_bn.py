"""E2cs k=15 path-support, BatchNorm-statistics, and training-log diagnostics.

Diagnostic only: never train, update checkpoint weights, or rerun the million-path
OOS evaluation. If original pnl.npz exists, use --pnl to measure its exact
200k saved visual-state sample; otherwise simulate an explicitly NEW Heston
sample with the same market parameters.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from scipy.interpolate import RegularGridInterpolator
from src.common.config import load_config
from src.common.seed import seed_everything
from src.replication.e1 import market_from_cfg, policy_from_cfg
from src.replication.e2cs import CallSpreadWorld, _model_delta_grids

ALPHAS = ("0.95", "0.99")


def training_diagnostics(log_path: Path) -> dict[str, Any]:
    """Window-mean training CVaR; never mislabeled as fixed validation."""
    pattern = re.compile(
        r"iter\s+(\d+)/\d+\s+lr=([0-9.eE+-]+)\s+"
        r"train_cvar=([0-9.eE+-]+)\s+\(last\s+(\d+)\s+avg\s+([0-9.eE+-]+)\)"
    )
    lines = log_path.read_text(encoding="utf-8").splitlines()
    result: dict[str, Any] = {a: [] for a in ALPHAS}
    active: str | None = None
    for line in lines:
        if "training call-spread policy alpha=0.95" in line:
            active = "0.95"
        elif "training call-spread policy alpha=0.99" in line:
            active = "0.99"
        match = pattern.search(line)
        if match and active:
            step, lr, last_batch, nwindow, mean = match.groups()
            result[active].append(
                {
                    "step": int(step),
                    "lr": float(lr),
                    "last_batch_cvar": float(last_batch),
                    "window_n": int(nwindow),
                    "mean_training_cvar": float(mean),
                }
            )
    summary: dict[str, Any] = {}
    for alpha in ALPHAS:
        series = result[alpha]
        if len(series) < 4 or series[-1]["step"] != 30000:
            raise ValueError(f"{log_path}: incomplete 30k-iteration log for alpha={alpha}")
        means = np.array([entry["mean_training_cvar"] for entry in series])
        summary[alpha] = {
            "series": series,
            "early_window_mean": float(means[0]),
            "final_window_mean": float(means[-1]),
            "last_3_window_mean": float(np.mean(means[-3:])),
            "previous_3_window_mean": float(np.mean(means[-6:-3])),
            "last_3_vs_previous_3_change": float(np.mean(means[-3:]) - np.mean(means[-6:-3])),
            "final_last_batch_minus_window": float(series[-1]["last_batch_cvar"] - means[-1]),
            "has_fixed_validation_history": False,
        }
    return summary


def _load_policies(checkpoint_root: Path, seed: int, device: torch.device):
    policies = {}
    for alpha in ALPHAS:
        folder = checkpoint_root / alpha / str(seed)
        config = load_config(str(folder / "config.yaml"))
        if config["policy"].get("recurrent", True) or config["policy"].get("norm") != "batch":
            raise ValueError(f"{folder}: expected corrected simpler policy with BatchNorm")
        market = market_from_cfg(config)
        policy = policy_from_cfg(
            config, market.n_features, market.d, device, recurrent=False, n_features_override=2
        )
        policy.load_state_dict(
            torch.load(folder / "policy.pt", map_location=device, weights_only=True),
            strict=True,
        )
        policy.eval()
        policies[alpha] = policy
    return policies


def _sample_states(
    cfg: dict[str, Any],
    count: int,
    chunk: int,
    seed: int,
    step: int,
    pnl: Path | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Use original saved visual spots/variance if supplied; else fresh simulation."""
    if pnl is not None:
        if not pnl.is_file():
            raise FileNotFoundError(pnl)
        with np.load(pnl, allow_pickle=False) as archive:
            spots = np.asarray(archive["visual_spot"], dtype=np.float64)
            vars_ = np.asarray(archive["visual_variance"], dtype=np.float64)
        source = {
            "kind": "original_e2cs_visual_npz",
            "file": str(pnl),
            "note": "Original saved k=15 visual states; not newly simulated.",
        }
    else:
        if count < 100 or chunk < 1:
            raise ValueError("Need >=100 fresh paths, batch >=1")
        seed_everything(seed)
        m = market_from_cfg(cfg)
        world = CallSpreadWorld(
            m,
            float(cfg["payoff"]["strike1"]),
            float(cfg["payoff"]["strike2"]),
            device="cpu",
        )
        s_parts, v_parts = [], []
        for start in range(0, count, chunk):
            paths = world.simulate(min(chunk, count - start))
            s_parts.append(paths.spot[:, step].detach().cpu().numpy())
            v_parts.append(paths.variance[:, step].detach().cpu().numpy())
        spots = np.concatenate(s_parts)
        vars_ = np.concatenate(v_parts)
        source = {
            "kind": "fresh_independent_cpu_heston_simulation",
            "seed": seed,
            "device": "cpu",
            "note": (
                "Same E2cs Heston setup, different CPU random stream from the "
                "archived GPU OOS visual paths; not an exact OOS sample reproduction."
            ),
        }
    if spots.shape != vars_.shape or spots.ndim != 1:
        raise ValueError("Expected paired 1D spot/variance vectors")
    if not (np.isfinite(spots).all() and np.isfinite(vars_).all()):
        raise ValueError("Non-finite simulated market state")
    return spots, vars_, source


def _batchnorm_diagnostics(
    step_net: torch.nn.Module, inputs: torch.Tensor, out_scale: torch.Tensor
) -> dict[str, Any]:
    """Compare running moments with current state-distribution moments.

    A batch-statistics counterfactual is a diagnostic ONLY. It is not an
    alternative OOS score and does not alter the persisted checkpoint.
    """
    step_net.eval()
    with torch.no_grad():
        z1 = step_net.fc1(inputs)
        z2 = step_net.fc2(torch.relu(step_net.bn1(z1)))
        evaluation = step_net(inputs) * out_scale
        diagnostics = {}
        for layer, pre in (("bn1", z1), ("bn2", z2)):
            bn = getattr(step_net, layer)
            running_mean = bn.running_mean
            running_var = bn.running_var
            if running_mean is None or running_var is None:
                raise ValueError("BatchNorm checkpoint has no running statistics")
            live_mean = pre.mean(dim=0)
            live_var = pre.var(dim=0, unbiased=False)
            mean_shift = (live_mean - running_mean).abs() / torch.sqrt(running_var + bn.eps)
            variance_ratio = live_var / (running_var + bn.eps)
            diagnostics[layer] = {
                "num_batches_tracked": int(bn.num_batches_tracked),
                "running_var_min": float(running_var.min()),
                "running_var_median": float(running_var.median()),
                "live_var_median": float(live_var.median()),
                "standardized_mean_shift_median": float(mean_shift.median()),
                "standardized_mean_shift_max": float(mean_shift.max()),
                "variance_ratio_median": float(variance_ratio.median()),
                "variance_ratio_min": float(variance_ratio.min()),
                "variance_ratio_max": float(variance_ratio.max()),
            }
        # Explicit functional BN with current-sample statistics; no mutation of
        # num_batches_tracked or running mean/variance.
        y1 = torch.relu(
            functional.batch_norm(
                z1,
                None,
                None,
                step_net.bn1.weight,
                step_net.bn1.bias,
                training=True,
                eps=step_net.bn1.eps,
            )
        )
        z2_batch = step_net.fc2(y1)
        y2 = torch.relu(
            functional.batch_norm(
                z2_batch,
                None,
                None,
                step_net.bn2.weight,
                step_net.bn2.bias,
                training=True,
                eps=step_net.bn2.eps,
            )
        )
        current_batch = step_net.out(y2) * out_scale
        delta = current_batch - evaluation
        diagnostics["batch_statistics_counterfactual"] = {
            "mean_abs_change_stock": float(delta[:, 0].abs().mean()),
            "mean_abs_change_varswap": float(delta[:, 1].abs().mean()),
            "rmse_change_stock": float(delta[:, 0].square().mean().sqrt()),
            "rmse_change_varswap": float(delta[:, 1].square().mean().sqrt()),
            "note": "Sensitivity to BN moment choice, not proof of a BN defect.",
        }
    return diagnostics


def _q(data: np.ndarray) -> dict[str, float]:
    probs = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    return {f"q{int(q * 100):02d}": float(np.quantile(data, q)) for q in probs}


def support_metrics(
    spots: np.ndarray, variances: np.ndarray, grid_spot: np.ndarray, grid_var: np.ndarray
) -> dict[str, Any]:
    """Proportion of real simulated states supported by Figures 8/9's grid."""
    in_s = (spots >= grid_spot[0]) & (spots <= grid_spot[-1])
    in_v = (variances >= grid_var[0]) & (variances <= grid_var[-1])
    mask = in_s & in_v
    # Explicit local zones have fixed, interpretable geometric boundaries.
    model_peak_zone = (spots >= 99.3) & (spots <= 100.3) & (variances >= 0.02) & (variances <= 0.04)
    shifted_peak_zone = (
        (spots >= 96.3) & (spots <= 97.3) & (variances >= 0.02) & (variances <= 0.04)
    )
    return {
        "n_paths": len(spots),
        "spot_quantiles": _q(spots),
        "variance_quantiles": _q(variances),
        "fraction_inside_spot_range": float(in_s.mean()),
        "fraction_inside_variance_range": float(in_v.mean()),
        "fraction_variance_exact_zero": float(np.mean(variances == 0.0)),
        "fraction_variance_below_grid": float(np.mean(variances < grid_var[0])),
        "fraction_variance_above_grid": float(np.mean(variances > grid_var[-1])),
        "fraction_inside_both_ranges": float(mask.mean()),
        "fraction_outside_both_ranges": float((~mask).mean()),
        "fraction_in_model_peak_neighborhood": float(model_peak_zone.mean()),
        "fraction_in_shifted_peak_neighborhood": float(shifted_peak_zone.mean()),
        "fraction_in_S_99_to_101": float(((spots >= 99) & (spots <= 101)).mean()),
        "fraction_in_V_0p02_to_0p04": float(((variances >= 0.02) & (variances <= 0.04)).mean()),
    }


def _weighted_comparison(
    spots: np.ndarray,
    vars_: np.ndarray,
    model_stock: np.ndarray,
    model_varswap: np.ndarray,
    grid_spot: np.ndarray,
    grid_var: np.ndarray,
    policies: dict[str, torch.nn.Module],
    step: int,
) -> dict[str, Any]:
    """Conditional error within the depicted grid, evaluated at sampled states."""
    mask = (
        (spots >= grid_spot[0])
        & (spots <= grid_spot[-1])
        & (vars_ >= grid_var[0])
        & (vars_ <= grid_var[-1])
    )
    inputs = torch.as_tensor(np.column_stack((spots, vars_)), dtype=torch.float64)
    valid = np.column_stack((spots[mask], vars_[mask]))
    stock_model = RegularGridInterpolator((grid_spot, grid_var), model_stock, bounds_error=True)(
        valid
    )
    var_model = RegularGridInterpolator((grid_spot, grid_var), model_varswap, bounds_error=True)(
        valid
    )
    model = np.column_stack((stock_model, var_model))
    spot_in = spots[mask]
    var_in = vars_[mask]
    zones: dict[str, np.ndarray] = {
        "model_peak_neighborhood": (
            (spot_in >= 99.3) & (spot_in <= 100.3) & (var_in >= 0.02) & (var_in <= 0.04)
        ),
        "shifted_peak_neighborhood": (
            (spot_in >= 96.3) & (spot_in <= 97.3) & (var_in >= 0.02) & (var_in <= 0.04)
        ),
    }
    for spot_name, s_lo, s_hi in (
        ("S95_99", 95.0, 99.0),
        ("S99_101", 99.0, 101.0),
        ("S101_103", 101.0, 103.0),
    ):
        for v_name, v_lo, v_hi in (
            ("V002_004", 0.02, 0.04),
            ("V004_008", 0.04, 0.08),
            ("V008_014", 0.08, 0.14),
        ):
            zones[f"{spot_name}_{v_name}"] = (
                (spot_in >= s_lo) & (spot_in < s_hi) & (var_in >= v_lo) & (var_in < v_hi)
            )
    out = {}
    for alpha, policy in policies.items():
        with torch.no_grad():
            positions = policy.steps[step](inputs) * policy.out_scale
        values = positions.detach().cpu().numpy()
        selected = values[mask]
        diff = selected - model
        per_leg = {}
        for i, leg in enumerate(("stock", "variance_swap")):
            per_leg[leg] = {
                "mean_signed_difference": float(diff[:, i].mean()),
                "rmse": float(np.sqrt(np.mean(diff[:, i] ** 2))),
                "mae": float(np.mean(np.abs(diff[:, i]))),
                "fraction_below_model": float(np.mean(diff[:, i] < 0)),
                "relative_to_uniform_grid": (
                    "Conditional to states within the paper figure grid; model delta "
                    "is bilinearly interpolated, not re-priced at each state."
                ),
            }
        regional = {}
        squared = diff**2
        for zone_name, zone_mask in zones.items():
            n_zone = int(np.sum(zone_mask))
            regional[zone_name] = {
                "n_paths": n_zone,
                "fraction_of_all_paths": n_zone / len(spots),
                "fraction_of_in_grid_paths": n_zone / len(valid),
            }
            for i, leg in enumerate(("stock", "variance_swap")):
                total_ss = float(np.sum(squared[:, i]))
                zone_ss = float(np.sum(squared[zone_mask, i]))
                regional[zone_name][leg] = {
                    "rmse": float(np.sqrt(np.mean(squared[zone_mask, i]))) if n_zone else None,
                    "mean_signed_difference": float(np.mean(diff[zone_mask, i]))
                    if n_zone
                    else None,
                    "share_of_in_grid_squared_error": zone_ss / total_ss if total_ss > 0 else None,
                }
        out[alpha] = {
            "regional_errors_within_grid": regional,
            "path_weighted_within_grid": per_leg,
            "stock_actual_positions_quantiles": _q(values[:, 0]),
            "varswap_actual_positions_quantiles": _q(values[:, 1]),
            "batchnorm": _batchnorm_diagnostics(policy.steps[step], inputs, policy.out_scale),
        }
    return out


def _figures(
    path: Path,
    spots: np.ndarray,
    vars_: np.ndarray,
    grid_spot: np.ndarray,
    grid_var: np.ndarray,
    fit: dict[str, Any],
    training: dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 6.3), constrained_layout=True)
    heat = ax.hist2d(
        spots,
        vars_,
        bins=(80, 80),
        range=((grid_spot[0], grid_spot[-1]), (0.0, max(0.2, float(np.quantile(vars_, 0.99))))),
        cmap="viridis",
        cmin=1,
    )[3]
    fig.colorbar(heat, ax=ax, label="Fresh k=15 simulated paths per bin")
    ax.axvline(99.8, color="white", linestyle="--", linewidth=1, label="Model stock peak S=99.8")
    ax.axvline(96.8, color="orange", linestyle=":", linewidth=1, label="99% NN stock peak S=96.8")
    ax.axhspan(0.02, 0.14, alpha=0.08, color="white")
    ax.set(xlabel="Spot price S at k=15", ylabel="Instantaneous variance V at k=15")
    ax.set_title(
        "E2cs path support (fresh independent Heston sample)\n"
        f"Figure-grid joint coverage = {fit['fraction_inside_both_ranges']:.1%}"
    )
    ax.legend(fontsize=8)
    fig.savefig(path / "e2cs_k15_path_support.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    for alpha, name in (("0.95", "CVaR 95%"), ("0.99", "CVaR 99%")):
        points = training[alpha]["series"]
        ax.plot(
            [q["step"] for q in points],
            [q["mean_training_cvar"] for q in points],
            marker="o",
            linewidth=1.6,
            label=name,
        )
    ax.set(xlabel="Training iteration", ylabel="Mean stochastic training CVaR")
    ax.set_title("E2cs training history (not fixed validation)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(path / "e2cs_training_cvar.png", dpi=200)
    plt.close(fig)


def analyze(
    *,
    checkpoint_root: Path,
    config_path: Path,
    output_dir: Path,
    figures_dir: Path,
    seed: int,
    step: int,
    n_paths: int,
    simulation_seed: int,
    batch_size: int,
    pnl: Path | None = None,
) -> dict[str, Any]:
    cfg = load_config(str(config_path))
    policies = _load_policies(checkpoint_root, seed, torch.device("cpu"))
    grid_spot = np.linspace(95, 103, 81)
    grid_var = np.linspace(0.02, 0.14, 61)
    market = market_from_cfg(cfg)
    model_stock, model_varswap = _model_delta_grids(
        market,
        float(cfg["payoff"]["strike1"]),
        float(cfg["payoff"]["strike2"]),
        step,
        grid_spot,
        grid_var,
    )
    spots, vars_, provenance = _sample_states(
        cfg,
        n_paths,
        batch_size,
        simulation_seed,
        step,
        pnl,
    )
    support = support_metrics(spots, vars_, grid_spot, grid_var)
    if support["fraction_inside_both_ranges"] == 0:
        raise ValueError("No sampled market states fall within the E2cs figure grid")
    weighted = _weighted_comparison(
        spots,
        vars_,
        model_stock,
        model_varswap,
        grid_spot,
        grid_var,
        policies,
        step,
    )
    training = training_diagnostics(checkpoint_root / "seed_0_master.log")
    report = {
        "seed": seed,
        "k": step,
        "sampling": provenance,
        "market_support": support,
        "policy_comparison": weighted,
        "training": training,
        "limitations": [
            "Fresh CPU Heston paths are not the same as the original GPU OOS visual paths.",
            "Errors are conditioned on the plotted state grid; not full unconditional OOS.",
            "Model delta on simulated states uses bilinear interpolation of the plotted grid.",
            "Running-BN vs current-sample moment differences do not prove training mistakes.",
            "Training log has no fixed-validation history, so cannot certify convergence.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "e2cs_support_bn_training.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _figures(figures_dir, spots, vars_, grid_spot, grid_var, support, training)
    print(f"E2cs diagnostic sample: {len(spots)} k={step} {provenance['kind']}")
    print(f"Figure grid coverage: {support['fraction_inside_both_ranges']:.4f}")
    for alpha in ALPHAS:
        d = weighted[alpha]
        bn = d["batchnorm"]
        print(
            f"alpha {alpha}: conditional stock RMSE={d['path_weighted_within_grid']['stock']['rmse']:.5f}, "
            f"varswap RMSE={d['path_weighted_within_grid']['variance_swap']['rmse']:.3f}, "
            f"BN1 median mean shift={bn['bn1']['standardized_mean_shift_median']:.3f}, "
            f"BN2={bn['bn2']['standardized_mean_shift_median']:.3f}"
        )
    print(f"Saved diagnostics to {output_dir}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("backups/e2cs_simple_seed0/e2cs")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/exp/e2cs.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/e2cs"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/e2cs_diagnostics"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--n-paths", type=int, default=40000)
    parser.add_argument("--simulation-seed", type=int, default=928371)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument(
        "--pnl",
        type=Path,
        default=None,
        help="Optional actual archived 0.95/0/pnl.npz (visual_spot/variance)",
    )
    args = parser.parse_args()
    analyze(
        checkpoint_root=args.checkpoint_root,
        config_path=args.config,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        seed=args.seed,
        step=args.step,
        n_paths=args.n_paths,
        simulation_seed=args.simulation_seed,
        batch_size=args.batch_size,
        pnl=args.pnl,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
