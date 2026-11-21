"""E2 call-spread experiment: paired CVaR policies on common QE-M paths.

This module deliberately subclasses HestonWorld rather than changing its shared
European-call default. It is therefore safe to add to the current A2 repository
without altering E1/E2 behaviour.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch

from src.common.metrics import cvar
from src.common.seed import seed_everything
from src.replication.e1 import market_from_cfg, train_policy
from src.replication.env_heston import HestonMarket, HestonPaths, HestonWorld
from src.replication.pricing import HestonParams, heston_call_prices


class CallSpreadWorld(HestonWorld):
    """Heston world with the normalized bounded call-spread payoff.

    The E2CS paper-style simpler policy observes the current (S_k, V_k)
    state directly, rather than the generic three-feature HestonWorld state.
    """

    def features(self, paths: HestonPaths) -> torch.Tensor:
        n = self.market.n_steps
        return torch.stack(
            [paths.spot[:, :n], paths.variance[:, :n]],
            dim=-1,
        )

    def __init__(self, market: HestonMarket, strike1: float, strike2: float, **kwargs: Any):
        if not strike2 > strike1:
            raise ValueError(f"call-spread requires strike2 > strike1; got {strike1}, {strike2}")
        super().__init__(market, **kwargs)
        self.strike1 = float(strike1)
        self.strike2 = float(strike2)

    def payoff(self, paths: HestonPaths) -> torch.Tensor:
        terminal = paths.spot[:, -1]
        width = self.strike2 - self.strike1
        return (
            torch.clamp(terminal - self.strike1, min=0.0)
            - torch.clamp(terminal - self.strike2, min=0.0)
        ) / width


def _market_params(market: HestonMarket, strike: float) -> HestonParams:
    return HestonParams(
        kappa=market.kappa,
        theta=market.theta,
        sigma=market.sigma,
        rho=market.rho,
        v0=market.v0,
        s0=market.s0,
        strike=strike,
        maturity=market.maturity,
    )


def _model_delta_grids(
    market: HestonMarket,
    strike1: float,
    strike2: float,
    step: int,
    spot_grid: np.ndarray,
    variance_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference model hedge in (S,V), including the variance-swap leg.

    The stock delta is dC_spread/dS. The variance-swap delta is
    (dC_spread/dV)/(dS2/dV), using the paper's total-variance swap mark.
    """
    sg, vg = np.meshgrid(spot_grid, variance_grid, indexing="ij")
    tau = market.maturity - step * market.dt
    hs = 0.05
    hv = 2.0e-4

    def prices(strike: float, s: np.ndarray, v: np.ndarray) -> np.ndarray:
        return heston_call_prices(s, v, tau, _market_params(market, strike))

    c1_s = (prices(strike1, sg + hs, vg) - prices(strike1, sg - hs, vg)) / (2.0 * hs)
    c2_s = (prices(strike2, sg + hs, vg) - prices(strike2, sg - hs, vg)) / (2.0 * hs)
    c1_v = (prices(strike1, sg, vg + hv) - prices(strike1, sg, np.maximum(vg - hv, 1.0e-8))) / (
        vg + hv - np.maximum(vg - hv, 1.0e-8)
    )
    c2_v = (prices(strike2, sg, vg + hv) - prices(strike2, sg, np.maximum(vg - hv, 1.0e-8))) / (
        vg + hv - np.maximum(vg - hv, 1.0e-8)
    )
    width = strike2 - strike1
    stock_delta = (c1_s - c2_s) / width
    variance_derivative = (c1_v - c2_v) / width
    factor = (1.0 - np.exp(-market.kappa * tau)) / market.kappa
    varswap_delta = variance_derivative / factor
    return stock_delta, varswap_delta


def _policy_stock_grid(
    policy: torch.nn.Module,
    spot_grid: np.ndarray,
    variance_grid: np.ndarray,
    step: int,
    device: torch.device,
) -> np.ndarray:
    """Evaluate the simpler policy directly on the paper's (S,V) grid."""
    ss, vv = np.meshgrid(spot_grid, variance_grid, indexing="ij")
    state = torch.as_tensor(
        np.column_stack([ss.ravel(), vv.ravel()]),
        dtype=torch.float64,
        device=device,
    )
    policy.eval()
    with torch.no_grad():
        # recurrent=False: each date has its own F_k(S_k,V_k), no previous delta.
        output = policy.steps[step](state) * policy.out_scale
    return output[:, 0].reshape(ss.shape).cpu().numpy()


def _eval_pair(
    world: CallSpreadWorld,
    policy95: torch.nn.Module,
    policy99: torch.nn.Module,
    n_paths: int,
    batch_size: int,
    visual_paths: int,
    step: int,
    eval_seed: int,
    log: logging.Logger,
) -> dict[str, np.ndarray]:
    seed_everything(eval_seed)
    policy95.eval()
    policy99.eval()
    losses95: list[np.ndarray] = []
    losses99: list[np.ndarray] = []
    visual_s: list[np.ndarray] = []
    visual_v: list[np.ndarray] = []
    visual_d95: list[np.ndarray] = []
    visual_d99: list[np.ndarray] = []
    visual_vd95: list[np.ndarray] = []
    visual_vd99: list[np.ndarray] = []
    remaining_visual = visual_paths
    done = 0
    with torch.no_grad():
        while done < n_paths:
            size = min(batch_size, n_paths - done)
            paths = world.simulate(size)
            features = world.features(paths)
            pos95 = policy95.forward_path(features)
            pos99 = policy99.forward_path(features)
            loss95 = world.terminal_loss(pos95, paths)
            loss99 = world.terminal_loss(pos99, paths)
            losses95.append(loss95.detach().cpu().numpy())
            losses99.append(loss99.detach().cpu().numpy())
            take = min(size, remaining_visual)
            if take > 0:
                visual_s.append(paths.spot[:take, step].detach().cpu().numpy().astype(np.float32))
                visual_v.append(
                    paths.variance[:take, step].detach().cpu().numpy().astype(np.float32)
                )
                visual_d95.append(pos95[:take, step, 0].detach().cpu().numpy().astype(np.float32))
                visual_d99.append(pos99[:take, step, 0].detach().cpu().numpy().astype(np.float32))
                visual_vd95.append(pos95[:take, step, 1].detach().cpu().numpy().astype(np.float32))
                visual_vd99.append(pos99[:take, step, 1].detach().cpu().numpy().astype(np.float32))
                remaining_visual -= take
            done += size
            if done % max(batch_size * 4, batch_size) == 0 or done == n_paths:
                log.info("  OOS evaluation %d/%d paths", done, n_paths)
    return {
        "loss95": np.concatenate(losses95),
        "loss99": np.concatenate(losses99),
        "visual_spot": np.concatenate(visual_s),
        "visual_variance": np.concatenate(visual_v),
        "visual_stock_delta95": np.concatenate(visual_d95),
        "visual_stock_delta99": np.concatenate(visual_d99),
        "visual_varswap_delta95": np.concatenate(visual_vd95),
        "visual_varswap_delta99": np.concatenate(visual_vd99),
    }


def run_pair(
    cfg: dict[str, Any], seed: int, device: torch.device, log: logging.Logger
) -> dict[str, Any]:
    """Train alpha=.95/.99 policies and evaluate them on identical QE-M paths."""
    base_market = market_from_cfg(cfg)
    if base_market.d != 2:
        raise ValueError("E2 call spread requires Heston d=2 (stock + variance swap)")
    if base_market.eps != 0.0:
        raise ValueError("E2 call spread requires eps=0")
    k1 = float(cfg["payoff"].get("strike1", base_market.strike))
    k2 = float(cfg["payoff"].get("strike2", 101.0))
    world = CallSpreadWorld(base_market, k1, k2, device=device)
    train_cfg = cfg.get("train", {}) or {}
    alphas = (0.95, 0.99)
    policies: dict[float, torch.nn.Module] = {}
    for alpha in alphas:
        local = copy.deepcopy(cfg)
        local["alpha"] = alpha
        log.info("training call-spread policy alpha=%.2f", alpha)
        seed_everything(seed)
        policies[alpha] = train_policy(
            world,
            local,
            device,
            log,
            recurrent=False,
            n_features_override=2,
        )

    eval_paths = int(train_cfg.get("eval_paths", 1_000_000))
    eval_batch = int(train_cfg.get("eval_batch_size", 25_000))
    visual_paths = min(int(train_cfg.get("visual_paths", 200_000)), eval_paths)
    visual_step = int(train_cfg.get("visual_step", base_market.n_steps // 2))
    eval_seed = int(train_cfg.get("eval_seed", 314159))
    evaluated = _eval_pair(
        world,
        policies[0.95],
        policies[0.99],
        eval_paths,
        eval_batch,
        visual_paths,
        visual_step,
        eval_seed,
        log,
    )
    loss95 = torch.as_tensor(evaluated["loss95"], dtype=torch.float64)
    loss99 = torch.as_tensor(evaluated["loss99"], dtype=torch.float64)
    params1 = _market_params(base_market, k1)
    params2 = _market_params(base_market, k2)
    call1 = heston_call_prices(
        np.array([base_market.s0]), np.array([base_market.v0]), base_market.maturity, params1
    )[0]
    call2 = heston_call_prices(
        np.array([base_market.s0]), np.array([base_market.v0]), base_market.maturity, params2
    )[0]
    q = float((call1 - call2) / (k2 - k1))
    price95 = float(cvar(loss95, 0.95))
    price99 = float(cvar(loss99, 0.99))

    # Paper Figure 8–9 evaluate the stock hedge on a compact (S,V) state grid.
    spot_grid = np.linspace(95.0, 103.0, 81)
    variance_grid = np.linspace(0.02, 0.14, 61)
    model_stock, model_varswap = _model_delta_grids(
        base_market, k1, k2, visual_step, spot_grid, variance_grid
    )
    policy_stock95 = _policy_stock_grid(
        policies[0.95], spot_grid, variance_grid, visual_step, device
    )
    policy_stock99 = _policy_stock_grid(
        policies[0.99], spot_grid, variance_grid, visual_step, device
    )
    metrics = {
        "price_alpha95": price95,
        "price_alpha99": price99,
        "q_model_call_spread": q,
        "mean_loss_alpha95": float(loss95.mean()),
        "mean_loss_alpha99": float(loss99.mean()),
        "cvar95_alpha95": float(cvar(loss95, 0.95)),
        "cvar99_alpha95": float(cvar(loss95, 0.99)),
        "cvar95_alpha99": float(cvar(loss99, 0.95)),
        "cvar99_alpha99": float(cvar(loss99, 0.99)),
        "visual_step": float(visual_step),
        "visual_paths": float(visual_paths),
    }
    checks = {
        "finite_prices": bool(np.isfinite([price95, price99, q]).all()),
        "finite_policy_positions": bool(
            np.isfinite(evaluated["visual_stock_delta95"]).all()
            and np.isfinite(evaluated["visual_stock_delta99"]).all()
        ),
        "model_surface_finite": bool(
            np.isfinite(model_stock).all() and np.isfinite(model_varswap).all()
        ),
        "common_oos_path_count": len(evaluated["loss95"]) == len(evaluated["loss99"]) == eval_paths,
    }
    log.info(
        "E2CS eval q=%.5f price95=%.5f price99=%.5f | "
        "CVaR99(95-policy)=%.5f CVaR99(99-policy)=%.5f",
        q,
        price95,
        price99,
        metrics["cvar99_alpha95"],
        metrics["cvar99_alpha99"],
    )
    return {
        "metrics": metrics,
        "checks": checks,
        "n_paths": eval_paths,
        "payloads": {
            "0.95": {
                "alpha": 0.95,
                "terminal_loss": evaluated["loss95"],
                "extra_pnl": {
                    "visual_spot": evaluated["visual_spot"],
                    "visual_variance": evaluated["visual_variance"],
                    "visual_stock_delta": evaluated["visual_stock_delta95"],
                    "visual_stock_delta_other_alpha": evaluated["visual_stock_delta99"],
                    "visual_varswap_delta": evaluated["visual_varswap_delta95"],
                    "model_spot_grid": spot_grid,
                    "model_variance_grid": variance_grid,
                    "model_stock_delta_grid": model_stock,
                    "model_varswap_delta_grid": model_varswap,
                    "policy_stock_delta_grid": policy_stock95,
                },
                "policy": policies[0.95].state_dict(),
            },
            "0.99": {
                "alpha": 0.99,
                "terminal_loss": evaluated["loss99"],
                "extra_pnl": {
                    "visual_spot": evaluated["visual_spot"],
                    "visual_variance": evaluated["visual_variance"],
                    "visual_stock_delta": evaluated["visual_stock_delta99"],
                    "visual_stock_delta_other_alpha": evaluated["visual_stock_delta95"],
                    "visual_varswap_delta": evaluated["visual_varswap_delta99"],
                    "model_spot_grid": spot_grid,
                    "model_variance_grid": variance_grid,
                    "model_stock_delta_grid": model_stock,
                    "model_varswap_delta_grid": model_varswap,
                    "policy_stock_delta_grid": policy_stock99,
                },
                "policy": policies[0.99].state_dict(),
            },
        },
    }
