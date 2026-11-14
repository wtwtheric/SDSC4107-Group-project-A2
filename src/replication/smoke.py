"""G0 冒烟实验：GBM 假路径 + 常数仓位，只验证 CLI/配置/落盘/指标管线。

不是 Heston，也不产生任何可报告的结论；env_heston.py 落地后由 kind: heston 取代。
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from src.common.metrics import cvar, proportional_cost, turnover
from src.policy.policy import ConstantPolicy


def run(
    cfg: dict[str, Any],
    seed: int,
    device: torch.device,
    log: logging.Logger,
) -> dict[str, Any]:
    """跑一条 GBM 的 30 步路径批次，返回 metrics / checks / terminal_loss。"""
    market = cfg["market"]
    n_steps = int(cfg["n_steps"])
    batch = int(cfg["batch"])
    horizon = float(market["t"])
    sigma = float(market["sigma"])
    s0 = float(market["s0"])
    strike = float(market["k"])
    dtype = torch.float64

    torch.manual_seed(int(seed))
    dt = horizon / n_steps
    drift = -0.5 * sigma**2 * dt
    shock = sigma * math.sqrt(dt)
    z = torch.randn(batch, n_steps, device=device, dtype=dtype)
    increments = drift + shock * z
    log_paths = torch.cat(
        [torch.zeros(batch, 1, device=device, dtype=dtype), increments.cumsum(dim=1)],
        dim=1,
    )
    prices = s0 * log_paths.exp()

    policy = ConstantPolicy(d=1, value=float(cfg["policy"]["value"]))
    policy.reset()
    positions = torch.empty(batch, n_steps, 1, device=device, dtype=dtype)
    for k in range(n_steps):
        x_k = torch.stack(
            [
                torch.log(prices[:, k] / s0),
                torch.full((batch,), k / n_steps, device=device, dtype=dtype),
            ],
            dim=-1,
        )
        positions[:, k] = policy.forward(x_k)

    cost = proportional_cost(positions, prices.unsqueeze(-1), float(cfg["eps"]))
    price_steps = prices.unsqueeze(-1)
    hedge = (positions * (price_steps[:, 1:] - price_steps[:, :-1])).sum(dim=(1, 2))
    payoff = torch.clamp(prices[:, -1] - strike, min=0.0)
    terminal_loss = payoff - hedge + cost

    metrics = {
        "mean_loss": float(terminal_loss.mean()),
        "sd_loss": float(terminal_loss.std()),
        "cvar50": float(cvar(terminal_loss, 0.5)),
        "cvar99": float(cvar(terminal_loss, 0.99)),
        "payoff_mean": float(payoff.mean()),
        "hedge_mean": float(hedge.mean()),
        "cost_mean": float(cost.mean()),
        "turnover_mean": float(turnover(positions).sum(dim=-1).mean()),
    }
    checks = {
        "positions_shape_ok": tuple(positions.shape) == (batch, n_steps, 1),
        "cost_nonnegative": bool(cost.min() >= 0.0),
        "cvar_monotone": metrics["cvar99"] >= metrics["cvar50"],
    }
    log.info(
        "smoke mean_loss=%.4f cvar50=%.4f cvar99=%.4f cost_mean=%.4f",
        metrics["mean_loss"],
        metrics["cvar50"],
        metrics["cvar99"],
        metrics["cost_mean"],
    )
    return {
        "metrics": metrics,
        "checks": checks,
        "n_paths": batch,
        "terminal_loss": terminal_loss.detach().cpu(),
    }
