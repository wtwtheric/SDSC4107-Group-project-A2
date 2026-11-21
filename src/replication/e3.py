"""E3: proportional transaction costs, recurrent versus simpler policy."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from src.common.metrics import cvar, proportional_cost
from src.common.seed import seed_everything
from src.replication.e1 import market_from_cfg, q_price, train_policy
from src.replication.env_heston import HestonWorld


def run(
    cfg: dict[str, Any], seed: int, device: torch.device, log: logging.Logger
) -> dict[str, Any]:
    market = market_from_cfg(cfg)
    if market.d != 2:
        raise ValueError("E3 requires Heston d=2")
    if market.eps <= 0.0:
        raise ValueError("E3 requires positive proportional transaction costs")
    alpha = float(cfg.get("alpha", 0.99))
    train_cfg = cfg.get("train", {}) or {}
    world = HestonWorld(market, device=device)

    log.info("training E3 recurrent policy alpha=%.3f eps=%.4g", alpha, market.eps)
    seed_everything(seed)
    recurrent = train_policy(world, cfg, device, log, recurrent=True)
    log.info("training E3 simpler policy alpha=%.3f eps=%.4g", alpha, market.eps)
    seed_everything(seed)
    simpler = train_policy(world, cfg, device, log, recurrent=False)

    n_paths = int(train_cfg.get("eval_paths", 1_000_000))
    batch_size = int(train_cfg.get("eval_batch_size", 25_000))
    eval_seed = int(train_cfg.get("eval_seed", 314159))
    visual_step = int(train_cfg.get("visual_step", market.n_steps // 2))
    seed_everything(eval_seed)
    recurrent_losses: list[np.ndarray] = []
    simpler_losses: list[np.ndarray] = []
    r_costs: list[np.ndarray] = []
    s_costs: list[np.ndarray] = []
    r_stock_costs: list[np.ndarray] = []
    r_var_costs: list[np.ndarray] = []
    s_stock_costs: list[np.ndarray] = []
    s_var_costs: list[np.ndarray] = []
    r_turnovers: list[np.ndarray] = []
    s_turnovers: list[np.ndarray] = []
    visual: dict[str, list[np.ndarray]] = {
        "spot": [], "variance": [], "r_stock": [], "s_stock": [], "r_varswap": [], "s_varswap": []
    }
    remaining_visual = min(int(train_cfg.get("visual_paths", 200_000)), n_paths)
    recurrent.eval()
    simpler.eval()
    done = 0
    with torch.no_grad():
        while done < n_paths:
            size = min(batch_size, n_paths - done)
            paths = world.simulate(size)
            features = world.features(paths)
            rp = recurrent.forward_path(features)
            sp = simpler.forward_path(features)
            rl = world.terminal_loss(rp, paths)
            sl = world.terminal_loss(sp, paths)
            recurrent_losses.append(rl.cpu().numpy())
            simpler_losses.append(sl.cpu().numpy())
            prices = world.prices(paths)
            r_cost = world.cost(rp, paths)
            s_cost = world.cost(sp, paths)
            r_stock_cost = proportional_cost(rp[..., :1], prices[..., :1], market.eps)
            r_var_cost = proportional_cost(rp[..., 1:2], prices[..., 1:2], market.eps)
            s_stock_cost = proportional_cost(sp[..., :1], prices[..., :1], market.eps)
            s_var_cost = proportional_cost(sp[..., 1:2], prices[..., 1:2], market.eps)
            r_costs.append(r_cost.cpu().numpy())
            s_costs.append(s_cost.cpu().numpy())
            r_stock_costs.append(r_stock_cost.cpu().numpy())
            r_var_costs.append(r_var_cost.cpu().numpy())
            s_stock_costs.append(s_stock_cost.cpu().numpy())
            s_var_costs.append(s_var_cost.cpu().numpy())
            r_prev = torch.cat([torch.zeros_like(rp[:, :1]), rp[:, :-1]], dim=1)
            s_prev = torch.cat([torch.zeros_like(sp[:, :1]), sp[:, :-1]], dim=1)
            r_turnovers.append((rp - r_prev).abs().sum(dim=(1, 2)).cpu().numpy())
            s_turnovers.append((sp - s_prev).abs().sum(dim=(1, 2)).cpu().numpy())
            take = min(size, remaining_visual)
            if take:
                visual["spot"].append(
                    paths.spot[:take, visual_step].cpu().numpy().astype(np.float32)
                )
                visual["variance"].append(
                    paths.variance[:take, visual_step].cpu().numpy().astype(np.float32)
                )
                visual["r_stock"].append(rp[:take, visual_step, 0].cpu().numpy().astype(np.float32))
                visual["s_stock"].append(sp[:take, visual_step, 0].cpu().numpy().astype(np.float32))
                visual["r_varswap"].append(
                    rp[:take, visual_step, 1].cpu().numpy().astype(np.float32)
                )
                visual["s_varswap"].append(
                    sp[:take, visual_step, 1].cpu().numpy().astype(np.float32)
                )
                remaining_visual -= take
            done += size
            if done % max(batch_size * 4, batch_size) == 0 or done == n_paths:
                log.info("  E3 common OOS evaluation %d/%d", done, n_paths)

    lr = torch.as_tensor(np.concatenate(recurrent_losses), dtype=torch.float64)
    ls = torch.as_tensor(np.concatenate(simpler_losses), dtype=torch.float64)
    metrics = {
        "price_recurrent": float(cvar(lr, alpha)),
        "price_simpler": float(cvar(ls, alpha)),
        "cvar_recurrent": float(cvar(lr, alpha)),
        "cvar_simpler": float(cvar(ls, alpha)),
        "mean_loss_recurrent": float(lr.mean()),
        "mean_loss_simpler": float(ls.mean()),
        "mean_cost_recurrent": float(np.concatenate(r_costs).mean()),
        "mean_cost_simpler": float(np.concatenate(s_costs).mean()),
        "mean_stock_cost_recurrent": float(np.concatenate(r_stock_costs).mean()),
        "mean_varswap_cost_recurrent": float(np.concatenate(r_var_costs).mean()),
        "mean_stock_cost_simpler": float(np.concatenate(s_stock_costs).mean()),
        "mean_varswap_cost_simpler": float(np.concatenate(s_var_costs).mean()),
        "mean_turnover_recurrent": float(np.concatenate(r_turnovers).mean()),
        "mean_turnover_simpler": float(np.concatenate(s_turnovers).mean()),
        "q_zero_cost_reference": float(q_price(world)),
        "visual_step": float(visual_step),
        "visual_paths": float(min(int(train_cfg.get("visual_paths", 200_000)), n_paths)),
    }
    checks = {
        "finite_prices": bool(
            np.isfinite([metrics["price_recurrent"], metrics["price_simpler"]]).all()
        ),
        "nonnegative_transaction_costs": bool(
            min(metrics["mean_cost_recurrent"], metrics["mean_cost_simpler"]) >= -1e-12
        ),
        "common_oos_path_count": len(lr) == len(ls) == n_paths,
    }
    log.info(
        "E3 result recurrent=%.5f simpler=%.5f | costs recurrent=%.5f simpler=%.5f",
        metrics["price_recurrent"], metrics["price_simpler"],
        metrics["mean_cost_recurrent"], metrics["mean_cost_simpler"]
    )
    return {
        "metrics": metrics,
        "checks": checks,
        "n_paths": n_paths,
        "terminal_loss": lr.numpy(),
        "extra_pnl": {
            "terminal_loss_simpler": ls.numpy(),
            "visual_spot": np.concatenate(visual["spot"]),
            "visual_variance": np.concatenate(visual["variance"]),
            "visual_stock_delta_recurrent": np.concatenate(visual["r_stock"]),
            "visual_stock_delta_simpler": np.concatenate(visual["s_stock"]),
            "visual_varswap_delta_recurrent": np.concatenate(visual["r_varswap"]),
            "visual_varswap_delta_simpler": np.concatenate(visual["s_varswap"]),
        },
        "artifacts": {
            "policy_recurrent.pt": recurrent.state_dict(),
            "policy_simpler.pt": simpler.state_dict(),
        },
    }
