"""E4: entropic-risk deep hedging for BS d=1 and Heston d=2."""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from src.common.seed import seed_everything
from src.replication.e1 import market_from_cfg, policy_from_cfg
from src.replication.env_bs import BSMarket, BSWorld
from src.replication.env_heston import HestonWorld
from src.replication.losses import entropic_loss
from src.replication.pricing import HestonParams, bs_call, heston_call_prices


def _make_world(cfg: dict[str, Any], eps: float, device: torch.device):
    exp = str(cfg["exp"])
    if exp == "e4bs":
        m = cfg["market"]
        market = BSMarket(
            sigma=float(m.get("sigma", 0.2)),
            s0=float(m.get("s0", 100.0)),
            strike=float(m.get("k", 100.0)),
            maturity=float(m.get("t", 30.0 / 365.0)),
            n_steps=int(cfg["n_steps"]),
            eps=float(eps),
        )
        return BSWorld(market, device=device)
    if exp == "e4h":
        market = replace(market_from_cfg(cfg), eps=float(eps))
        if market.d != 2:
            raise ValueError("E4-Heston requires d=2")
        return HestonWorld(market, device=device)
    raise ValueError(f"unsupported E4 experiment id: {exp}")


def _zero_cost_price(world: Any) -> float:
    m = world.market
    if isinstance(world, BSWorld):
        return float(bs_call(m.s0, m.strike, m.maturity, m.sigma))
    params = HestonParams(
        kappa=m.kappa, theta=m.theta, sigma=m.sigma, rho=m.rho,
        v0=m.v0, s0=m.s0, strike=m.strike, maturity=m.maturity,
    )
    return float(heston_call_prices(np.array([m.s0]), np.array([m.v0]), m.maturity, params)[0])


def run_one(
    cfg: dict[str, Any], seed: int, device: torch.device, log: logging.Logger, eps: float
) -> dict[str, Any]:
    """Train and evaluate one epsilon point; outer grid runner handles directories."""
    world = _make_world(cfg, eps, device)
    m = world.market
    train_cfg = cfg.get("train", {}) or {}
    lam = float(cfg.get("risk", {}).get("lambda", 1.0))
    if lam <= 0.0:
        raise ValueError("entropic risk lambda must be positive")
    seed_everything(seed)
    policy = policy_from_cfg(cfg, m.n_features, m.d, device, recurrent=True)
    n_iter = int(train_cfg.get("n_iter", 30_000))
    batch = int(cfg.get("batch", 1024))
    lr = float(train_cfg.get("lr", 0.005))
    lr_min = float(train_cfg.get("lr_min", 0.0005))
    schedule = str(train_cfg.get("lr_schedule", "cosine")).lower()
    log_every = int(train_cfg.get("log_every", 2500))
    if schedule not in {"cosine", "constant"}:
        raise ValueError(f"unknown lr_schedule={schedule!r}")
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    log.info(
        "E4 train exp=%s eps=%.8g lambda=%.4g batch=%d n_iter=%d",
        cfg["exp"], eps, lam, batch, n_iter,
    )
    running = 0.0
    window_count = 0
    for iteration in range(1, n_iter + 1):
        if schedule == "cosine" and n_iter > 1:
            fraction = (iteration - 1) / (n_iter - 1)
            lr_now = lr_min + 0.5 * (lr - lr_min) * (1.0 + np.cos(np.pi * fraction))
        else:
            lr_now = lr
        for group in optimizer.param_groups:
            group["lr"] = float(lr_now)
        policy.train()
        paths = world.simulate(batch)
        positions = policy.forward_path(world.features(paths))
        objective = entropic_loss(world.terminal_loss(positions, paths), lam)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        running += float(objective.detach())
        window_count += 1
        if iteration % log_every == 0 or iteration == n_iter:
            log.info(
                "  iter %d/%d lr=%.4g train_entropic=%.6f avg=%.6f",
                iteration, n_iter, lr_now, float(objective.detach()), running / window_count
            )
            running = 0.0
            window_count = 0

    eval_paths = int(train_cfg.get("eval_paths", 100_000))
    eval_batch = int(train_cfg.get("eval_batch_size", 10_000))
    if eval_paths < 2 or eval_batch < 1:
        raise ValueError("E4 eval_paths must be >=2 and eval_batch_size must be positive")
    eval_seed = int(train_cfg.get("eval_seed", 271828))
    seed_everything(eval_seed)
    policy.eval()
    losses: list[np.ndarray] = []
    done = 0
    with torch.no_grad():
        while done < eval_paths:
            size = min(eval_batch, eval_paths - done)
            paths = world.simulate(size)
            positions = policy.forward_path(world.features(paths))
            losses.append(world.terminal_loss(positions, paths).cpu().numpy())
            done += size
            if done % max(eval_batch * 5, eval_batch) == 0 or done == eval_paths:
                log.info("  E4 OOS evaluation %d/%d", done, eval_paths)
    terminal_loss = np.concatenate(losses)
    loss_tensor = torch.as_tensor(terminal_loss, dtype=torch.float64)
    p_eps = float(entropic_loss(loss_tensor, lam))
    p0 = _zero_cost_price(world)
    increment = p_eps - p0
    metrics = {
        "eps": float(eps),
        "lambda": lam,
        "price_eps": p_eps,
        "price0": p0,
        "price_increment": increment,
        "mean_loss": float(loss_tensor.mean()),
        "std_loss": float(loss_tensor.std(unbiased=True)),
        "eval_seed": float(eval_seed),
    }
    checks = {
        "finite_price": bool(np.isfinite([p_eps, p0, increment]).all()),
        "positive_price_increment": bool(increment > 0.0),
        "evaluation_path_count": len(terminal_loss) == eval_paths,
    }
    log.info("E4 eval eps=%.8g p_eps=%.8f p0=%.8f increment=%.8g", eps, p_eps, p0, increment)
    return {
        "metrics": metrics,
        "checks": checks,
        "n_paths": eval_paths,
        "terminal_loss": terminal_loss,
        "artifacts": {"policy.pt": policy.state_dict()},
    }
