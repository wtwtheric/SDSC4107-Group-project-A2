"""E2: Heston d=2, zero-cost comparison of 50%- and 99%-CVaR policies.

Paper target: Section 5.2, Figures 6-7.

The two policies are trained with the same architecture, optimizer budget, initial
seed, and simulated training-path sequence. They are then evaluated on one common
out-of-sample path set. The reported realized CVaRs are normalized to the common
risk-neutral premium q, matching the comparison in paper Figure 7.

Metric naming follows docs/spec.md:
- "own"  = policy trained at alpha = 0.99;
- "other" = comparison policy trained at alpha = 0.50.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

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

    # Pair the two training runs as tightly as practical: same initialization seed
    # and same simulated path sequence. Only the CVaR objective differs.
    log.info("training E2 comparison policy alpha=%.2f", alpha_other)
    seed_everything(seed)
    policy_other = train_policy(world, cfg_other, device, log, recurrent=True)

    log.info("training E2 primary policy alpha=%.2f", alpha_own)
    seed_everything(seed)
    policy_own = train_policy(world, cfg_own, device, log, recurrent=True)

    # One common held-out path set for both strategies (paper Figures 6-7 logic).
    seed_everything(seed + eval_offset)
    paths = world.simulate(eval_paths)
    ev_other = evaluate_on_paths(world, policy_other, paths)
    ev_own = evaluate_on_paths(world, policy_own, paths)

    loss_other = ev_other["loss"]
    loss_own = ev_own["loss"]
    q = q_price(world)

    price_other = float(cvar(loss_other, alpha_other))
    price_own = float(cvar(loss_own, alpha_own))

    # Figure 7 normalization: both strategies are evaluated as if only q were charged.
    normalized_other = loss_other - q
    normalized_own = loss_own - q

    cvar05_own = float(cvar(normalized_own, 0.50))
    cvar99_own = float(cvar(normalized_own, 0.99))
    cvar05_other = float(cvar(normalized_other, 0.50))
    cvar99_other = float(cvar(normalized_other, 0.99))

    metrics: dict[str, float] = {
        # Required by docs/spec.md for e2.
        "price": price_own,
        "cvar05_own": cvar05_own,
        "cvar99_own": cvar99_own,
        "cvar05_other": cvar05_other,
        "cvar99_other": cvar99_other,
        # Additional traceability fields.
        "q": q,
        "price_alpha50": price_other,
        "price_alpha99": price_own,
        "mean_raw_loss_alpha50": float(loss_other.mean()),
        "mean_raw_loss_alpha99": float(loss_own.mean()),
    }

    checks: dict[str, bool] = {
        "price99_band": 3.2 <= price_own <= 3.8,
        "price50_reference_band": 1.80 <= price_other <= 2.10,
        "cvar50_tradeoff": cvar05_own > cvar05_other,
        "cvar99_tradeoff": cvar99_own < cvar99_other,
        "mean_loss_alpha50_near_q": abs(float(loss_other.mean()) - q) < 0.02,
        "mean_loss_alpha99_near_q": abs(float(loss_own.mean()) - q) < 0.02,
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
        # Primary E2 terminal_loss is the alpha=.99 raw loss, consistent with spec §5.3.
        "terminal_loss": loss_own.cpu().numpy(),
        "extra_pnl": {
            "terminal_loss_alpha50": loss_other.cpu().numpy(),
            "terminal_loss_alpha99": loss_own.cpu().numpy(),
            "normalized_loss_alpha50": normalized_other.cpu().numpy(),
            "normalized_loss_alpha99": normalized_own.cpu().numpy(),
        },
        "artifacts": {
            "policy_alpha50.pt": policy_other.state_dict(),
            "policy_alpha99.pt": policy_own.state_dict(),
        },
    }
