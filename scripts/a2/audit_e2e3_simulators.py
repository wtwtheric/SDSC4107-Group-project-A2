"""Compare the baseline QE-M Heston simulator with the paper-style simulator.

This is a simulator-only diagnostic: no policy is trained. It is intentionally
separate from the E2/E3 fidelity-control runs so simulator distribution effects
can be inspected before attributing changes to the network input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from src.common.seed import resolve_device, seed_everything
from src.replication.e1 import q_price
from src.replication.env_heston import HestonMarket, HestonWorld


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "q001": float(np.quantile(values, 0.001)),
        "q01": float(np.quantile(values, 0.01)),
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "q999": float(np.quantile(values, 0.999)),
    }


@torch.no_grad()
def summarize(
    simulator: str,
    *,
    n_paths: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    market = HestonMarket(simulator=simulator, feature_set="current")
    world = HestonWorld(market, device=device)
    seed_everything(seed)

    terminal_spot: list[np.ndarray] = []
    terminal_variance: list[np.ndarray] = []
    mid_variance: list[np.ndarray] = []
    payoff: list[np.ndarray] = []
    mid_varswap: list[np.ndarray] = []
    terminal_varswap: list[np.ndarray] = []
    done = 0
    while done < n_paths:
        size = min(batch_size, n_paths - done)
        paths = world.simulate(size)
        terminal_spot.append(paths.spot[:, -1].cpu().numpy())
        terminal_variance.append(paths.variance[:, -1].cpu().numpy())
        mid_variance.append(paths.variance[:, 15].cpu().numpy())
        payoff.append(world.payoff(paths).cpu().numpy())
        mid_varswap.append(paths.varswap[:, 15].cpu().numpy())
        terminal_varswap.append(paths.varswap[:, -1].cpu().numpy())
        done += size

    s_t = np.concatenate(terminal_spot)
    v_t = np.concatenate(terminal_variance)
    v_mid = np.concatenate(mid_variance)
    z = np.concatenate(payoff)
    s2_mid = np.concatenate(mid_varswap)
    s2_t = np.concatenate(terminal_varswap)
    s2_0 = market.theta * market.maturity

    return {
        "simulator": simulator,
        "paths": n_paths,
        "seed": seed,
        "spot_terminal": {
            "mean": float(s_t.mean()),
            "std": float(s_t.std()),
            **_quantiles(s_t),
        },
        "variance_mid": {
            "mean": float(v_mid.mean()),
            "std": float(v_mid.std()),
            "zero_fraction": float(np.mean(v_mid == 0.0)),
            "below_1e_12_fraction": float(np.mean(v_mid < 1e-12)),
            "below_0_02_fraction": float(np.mean(v_mid < 0.02)),
            "above_0_14_fraction": float(np.mean(v_mid > 0.14)),
            **_quantiles(v_mid),
        },
        "variance_terminal": {
            "mean": float(v_t.mean()),
            "std": float(v_t.std()),
            "zero_fraction": float(np.mean(v_t == 0.0)),
            "below_1e_12_fraction": float(np.mean(v_t < 1e-12)),
            **_quantiles(v_t),
        },
        "payoff": {
            "mean": float(z.mean()),
            "std": float(z.std()),
            **_quantiles(z),
        },
        "variance_swap": {
            "s2_0": float(s2_0),
            "mid_mean": float(s2_mid.mean()),
            "mid_relative_drift": float(s2_mid.mean() / s2_0 - 1.0),
            "terminal_mean": float(s2_t.mean()),
            "terminal_relative_drift": float(s2_t.mean() / s2_0 - 1.0),
        },
        "analytic_q": float(q_price(world)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e2e3_fidelity/simulator_audit.json"),
    )
    args = parser.parse_args()
    device = resolve_device(args.device)
    if args.paths < 1 or args.batch_size < 1:
        raise ValueError("--paths and --batch-size must be positive")

    qem = summarize(
        "qem",
        n_paths=args.paths,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    paper_sbk = summarize(
        "paper_sbk",
        n_paths=args.paths,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    paper_q = float(paper_sbk["analytic_q"])
    s2_0 = float(paper_sbk["variance_swap"]["s2_0"])
    checks = {
        "paper_terminal_spot_martingale_within_2pct": (
            abs(float(paper_sbk["spot_terminal"]["mean"]) / 100.0 - 1.0) < 0.02
        ),
        "paper_payoff_mean_within_3pct_of_analytic_q": (
            abs(float(paper_sbk["payoff"]["mean"]) / paper_q - 1.0) < 0.03
        ),
        "paper_mid_variance_mean_within_5pct_of_theta": (
            abs(float(paper_sbk["variance_mid"]["mean"]) / 0.04 - 1.0) < 0.05
        ),
        "paper_mid_varswap_mean_within_3pct_of_s20": (
            abs(float(paper_sbk["variance_swap"]["mid_mean"]) / s2_0 - 1.0) < 0.03
        ),
        "paper_terminal_varswap_mean_within_3pct_of_s20": (
            abs(float(paper_sbk["variance_swap"]["terminal_mean"]) / s2_0 - 1.0) < 0.03
        ),
    }
    results = {
        "schema_version": "1.0",
        "purpose": "E2/E3 simulator-only distribution audit",
        "device": str(device),
        "qem": qem,
        "paper_sbk": paper_sbk,
        "sanity_checks": checks,
        "passed": all(checks.values()),
        "interpretation_note": (
            "paper_sbk uses exact CIR endpoint transitions and the simplified "
            "Broadie-Kaya approximation stated in Section 5.2.2; it is not the "
            "full exact integrated-variance Broadie-Kaya algorithm."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(f"saved {args.output}")
    for name in ("qem", "paper_sbk"):
        item = results[name]
        print(
            name,
            "payoff_mean=",
            round(item["payoff"]["mean"], 6),
            "V15_zero=",
            round(item["variance_mid"]["zero_fraction"], 6),
            "V15<.02=",
            round(item["variance_mid"]["below_0_02_fraction"], 6),
            "S2_mid_drift=",
            round(item["variance_swap"]["mid_relative_drift"], 6),
        )
    print("sanity_checks=", checks, "passed=", results["passed"])
    return 0 if results["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
