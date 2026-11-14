"""E2: Heston d=2, zero-cost comparison of 50%- and 99%-CVaR policies.

Paper target: Section 5.2, Figures 6-7. Both policies use one common held-out
path set. Figure 6 uses each policy's own indifference premium; Figure 7 uses
the common risk-neutral premium q. Loss is L = Z - hedge_gain, so the plotted
Figure 7 hedging error is q-L while CVaR is calculated on the upper tail of L-q.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch

from src.common.metrics import cvar
from src.common.seed import seed_everything
from src.replication.e1 import market_from_cfg, q_price, train_policy
from src.replication.env_heston import HestonPaths, HestonWorld
from src.replication.net_recurrent import SemiRecurrentPolicy


def config_with_alpha(cfg: dict[str, Any], alpha: float) -> dict[str, Any]:
    """Return an isolated config copy with the requested CVaR confidence level."""
    out = copy.deepcopy(cfg)
    out["alpha"] = float(alpha)
    return out


@torch.no_grad()
def evaluate_on_paths(
    world: HestonWorld,
    policy: SemiRecurrentPolicy,
    paths: HestonPaths,
) -> dict[str, torch.Tensor]:
    """Evaluate one frozen policy on pre-generated paths shared across E2 policies."""
    policy.eval()
    features = world.features(paths)
    positions = policy.forward_path(features)
    loss = world.terminal_loss(positions, paths)
    return {"positions": positions, "loss": loss}


def _positions_for_archive(positions: torch.Tensor) -> np.ndarray:
    """Keep pathwise positions for the 100k diagnostic run in compact float32 form."""
    return positions.detach().cpu().numpy().astype(np.float32, copy=False)


def run(
    cfg: dict[str, Any],
    seed: int,
    device: torch.device,
    log: logging.Logger,
) -> dict[str, Any]:
    """Train alpha=0.50 and alpha=0.99 policies and cross-evaluate on common paths."""
    market = market_from_cfg(cfg)
    if market.d != 2:
        raise ValueError(f"E2 requires Heston d=2, got d={market.d}")
    if market.eps != 0.0:
        raise ValueError(f"E2 Figures 6-7 require zero transaction cost, got eps={market.eps}")

    world = HestonWorld(market, device=device)
    train_cfg = cfg.get("train", {}) or {}
    alpha_own = float(cfg.get("alpha", 0.99))
    alpha_other = float(cfg.get("compare_alpha", 0.50))
    if abs(alpha_own - 0.99) > 1e-12:
        raise ValueError(f"E2 primary alpha must be 0.99, got {alpha_own}")
    if abs(alpha_other - 0.50) > 1e-12:
        raise ValueError(f"E2 comparison alpha must be 0.50, got {alpha_other}")

    eval_paths = int(train_cfg.get("eval_paths", 100_000))
    eval_offset = int(train_cfg.get("eval_seed_offset", 10_000))
    cfg_other = config_with_alpha(cfg, alpha_other)
    cfg_own = config_with_alpha(cfg, alpha_own)

    # Pair the two training runs: same initialization seed and simulated path sequence.
    log.info("training E2 comparison policy alpha=%.2f", alpha_other)
    seed_everything(seed)
    policy_other = train_policy(world, cfg_other, device, log, recurrent=True)

    log.info("training E2 primary policy alpha=%.2f", alpha_own)
    seed_everything(seed)
    policy_own = train_policy(world, cfg_own, device, log, recurrent=True)

    # One common held-out path set for both strategies.
    seed_everything(seed + eval_offset)
    paths = world.simulate(eval_paths)
    ev_other = evaluate_on_paths(world, policy_other, paths)
    ev_own = evaluate_on_paths(world, policy_own, paths)

    loss_other = ev_other["loss"]
    loss_own = ev_own["loss"]
    q = q_price(world)
    price_other = float(cvar(loss_other, alpha_other))
    price_own = float(cvar(loss_own, alpha_own))

    loss_above_q_other = loss_other - q
    loss_above_q_own = loss_own - q
    hedging_error_common_q_other = q - loss_other
    hedging_error_common_q_own = q - loss_own
    hedging_error_own_price_other = price_other - loss_other
    hedging_error_own_price_own = price_own - loss_own

    cvar05_own = float(cvar(loss_above_q_own, 0.50))
    cvar99_own = float(cvar(loss_above_q_own, 0.99))
    cvar05_other = float(cvar(loss_above_q_other, 0.50))
    cvar99_other = float(cvar(loss_above_q_other, 0.99))
    mean_abs_error_other = float(hedging_error_common_q_other.abs().mean())
    mean_abs_error_own = float(hedging_error_common_q_own.abs().mean())

    metrics: dict[str, float] = {
        "price": price_own,
        "cvar05_own": cvar05_own,
        "cvar99_own": cvar99_own,
        "cvar05_other": cvar05_other,
        "cvar99_other": cvar99_other,
        "q": q,
        "price_alpha50": price_other,
        "price_alpha99": price_own,
        "mean_raw_loss_alpha50": float(loss_other.mean()),
        "mean_raw_loss_alpha99": float(loss_own.mean()),
        "mean_abs_hedging_error_alpha50": mean_abs_error_other,
        "mean_abs_hedging_error_alpha99": mean_abs_error_own,
    }
    checks: dict[str, bool] = {
        "price99_band": 3.2 <= price_own <= 3.8,
        "price50_reference_band": 1.80 <= price_other <= 2.10,
        "cvar50_tradeoff": cvar05_own > cvar05_other,
        "cvar99_tradeoff": cvar99_own < cvar99_other,
        "mean_loss_alpha50_near_q": abs(float(loss_other.mean()) - q) < 0.02,
        "mean_loss_alpha99_near_q": abs(float(loss_own.mean()) - q) < 0.02,
        "price_translation_alpha50": abs(price_other - (q + cvar05_other)) < 1e-10,
        "price_translation_alpha99": abs(price_own - (q + cvar99_own)) < 1e-10,
    }

    log.info(
        "E2 eval q=%.4f p50=%.4f p99=%.4f | "
        "q-normalized CVaR50: alpha99=%.4f alpha50=%.4f | "
        "CVaR99: alpha99=%.4f alpha50=%.4f",
        q,
        price_other,
        price_own,
        cvar05_own,
        cvar05_other,
        cvar99_own,
        cvar99_other,
    )

    return {
        "metrics": metrics,
        "checks": checks,
        "n_paths": eval_paths,
        "terminal_loss": loss_own.detach().cpu().numpy(),
        "extra_pnl": {
            "terminal_loss_alpha50": loss_other.detach().cpu().numpy(),
            "terminal_loss_alpha99": loss_own.detach().cpu().numpy(),
            "loss_above_q_alpha50": loss_above_q_other.detach().cpu().numpy(),
            "loss_above_q_alpha99": loss_above_q_own.detach().cpu().numpy(),
            "hedging_error_common_q_alpha50": hedging_error_common_q_other.detach().cpu().numpy(),
            "hedging_error_common_q_alpha99": hedging_error_common_q_own.detach().cpu().numpy(),
            "hedging_error_own_price_alpha50": hedging_error_own_price_other.detach().cpu().numpy(),
            "hedging_error_own_price_alpha99": hedging_error_own_price_own.detach().cpu().numpy(),
            "positions_alpha50": _positions_for_archive(ev_other["positions"]),
            "positions_alpha99": _positions_for_archive(ev_own["positions"]),
        },
        "artifacts": {
            "policy_alpha50.pt": policy_other.state_dict(),
            "policy_alpha99.pt": policy_own.state_dict(),
        },
    }
