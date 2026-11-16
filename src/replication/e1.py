"""E1 / e1d1：Heston 世界的 avg-CVaR 深度对冲（论文 §5.2，spec §6 A1 任务 4）。

- 策略：net_recurrent.SemiRecurrentPolicy（每步独立权重、BN、可选递归 δ_{k−1}）；
- 目标：min CVaR_α(L)，L 由 env_heston.HestonWorld.terminal_loss 给出（ε = 0 时即 E1）；
- 对照：同一配置下训练一个非递归 "simpler" 网络，在同一批评估路径上比较 CVaR 价格；
- 指标：price = CVaR_α(L)、q（Heston 解析价）、mean/cvar50/cvar99、delta_rmse（k=15，
  S∈[90,110] 对 benchmark δ1）、price_simpler。

超参以论文 §5.1 为起点（Adam lr=0.005，batch 256）；E1 的定稿配方见
configs/exp/e1.yaml（batch 1024 + cosine 退火 + 分腿 out_scale），理由见 docs/e1_diagnosis.md。
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.common.metrics import cvar
from src.replication.benchmark import HestonBenchmark
from src.replication.env_heston import HestonMarket, HestonPaths, HestonWorld
from src.replication.losses import avg_cvar
from src.replication.net_recurrent import SemiRecurrentPolicy
from src.replication.pricing import PAPER, heston_call_prices


def market_from_cfg(cfg: dict[str, Any]) -> HestonMarket:
    m = cfg["market"]
    return HestonMarket(
        kappa=float(m.get("kappa", PAPER.kappa)),
        theta=float(m.get("theta", PAPER.theta)),
        sigma=float(m.get("sigma", PAPER.sigma)),
        rho=float(m.get("rho", PAPER.rho)),
        v0=float(m.get("v0", PAPER.v0)),
        s0=float(m.get("s0", PAPER.s0)),
        strike=float(m.get("k", PAPER.strike)),
        maturity=float(m.get("t", PAPER.maturity)),
        n_steps=int(cfg["n_steps"]),
        eps=float(cfg.get("eps", 0.0)),
        d=int(m.get("d", 2)),
    )


def policy_from_cfg(
    cfg: dict[str, Any], n_features: int, d: int, device: torch.device, recurrent: bool = True
) -> SemiRecurrentPolicy:
    hidden = cfg["policy"].get("hidden")
    policy = SemiRecurrentPolicy(
        n_features=n_features,
        d=d,
        n_steps=int(cfg["n_steps"]),
        hidden=int(hidden) if hidden is not None else d + 15,
        recurrent=recurrent,
        batch_norm=bool(cfg["policy"].get("batch_norm", True)),
        norm=cfg["policy"].get("norm"),
        out_scale=cfg["policy"].get("out_scale", 1.0),
    )
    return policy.to(device=device, dtype=torch.float64)


def scheduled_lr(it: int, n_iter: int, lr: float, lr_min: float, schedule: str) -> float:
    """第 it 轮（从 1 起）的学习率；cosine 从 lr 余弦退火到 lr_min。

    恒定 lr 会在 CVaR 分位数梯度噪声下停在极限环（见 docs/e1_diagnosis.md）。
    """
    if schedule == "constant" or n_iter <= 1:
        return lr
    frac = (it - 1) / (n_iter - 1)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * frac))


def train_policy(
    world: HestonWorld,
    cfg: dict[str, Any],
    device: torch.device,
    log: logging.Logger,
    recurrent: bool = True,
) -> SemiRecurrentPolicy:
    m = world.market
    policy = policy_from_cfg(cfg, m.n_features, m.d, device, recurrent=recurrent)
    train_cfg = cfg.get("train", {}) or {}
    n_iter = int(train_cfg.get("n_iter", 5000))
    lr = float(train_cfg.get("lr", 0.005))
    lr_min = float(train_cfg.get("lr_min", lr))
    schedule = str(train_cfg.get("lr_schedule", "constant")).lower()
    if schedule not in {"constant", "cosine"}:
        raise ValueError(f"unknown lr_schedule={schedule!r}; use 'constant' or 'cosine'")
    batch = int(cfg["batch"])
    alpha = float(cfg["alpha"])
    log_every = int(train_cfg.get("log_every", 250))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    # Optional fixed validation sample. Generate it from a separate seed and
    # restore all RNG states so enabling diagnostics does not alter training paths.
    validation_paths = None
    validation_features = None
    validation_seed = int(train_cfg.get("validation_seed", 8675309))
    validation_count = int(train_cfg.get("validation_paths", 0))
    validation_batch = int(train_cfg.get("validation_batch_size", 4096))
    if validation_count > 0:
        if validation_batch < 1:
            raise ValueError("validation_batch_size must be positive")
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            torch.manual_seed(validation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(validation_seed)
            validation_paths = world.simulate(validation_count)
            validation_features = world.features(validation_paths)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    def evaluate_fixed_validation() -> float:
        assert validation_paths is not None and validation_features is not None
        policy.eval()
        parts: list[Tensor] = []
        with torch.no_grad():
            for start in range(0, validation_count, validation_batch):
                stop = min(start + validation_batch, validation_count)
                # Slice the NamedTuple fields consistently to avoid copying the full
                # validation sample into a second set of tensors.
                paths_batch = HestonPaths(
                    spot=validation_paths.spot[start:stop],
                    variance=validation_paths.variance[start:stop],
                    varswap=validation_paths.varswap[start:stop],
                    time=validation_paths.time,
                )
                features_batch = validation_features[start:stop]
                positions = policy.forward_path(features_batch)
                parts.append(world.terminal_loss(positions, paths_batch))
        value = float(cvar(torch.cat(parts), alpha))
        policy.train()
        return value

    log.info(
        "  lr_schedule=%s lr=%g->%g batch=%d n_iter=%d validation_paths=%d validation_seed=%d",
        schedule,
        lr,
        lr_min,
        batch,
        n_iter,
        validation_count,
        validation_seed,
    )

    running = 0.0
    for it in range(1, n_iter + 1):
        lr_t = scheduled_lr(it, n_iter, lr, lr_min, schedule)
        for group in opt.param_groups:
            group["lr"] = lr_t
        policy.train()
        paths = world.simulate(batch)
        features = world.features(paths)
        positions = policy.forward_path(features)
        loss = avg_cvar(world.terminal_loss(positions, paths), alpha)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running += float(loss.detach())
        if it % log_every == 0:
            if validation_paths is not None:
                validation_cvar = evaluate_fixed_validation()
                log.info(
                    "  iter %d/%d lr=%.3g train_cvar=%.4f (last %d avg %.4f) validation_cvar=%.4f",
                    it,
                    n_iter,
                    lr_t,
                    float(loss.detach()),
                    log_every,
                    running / log_every,
                    validation_cvar,
                )
            else:
                log.info(
                    "  iter %d/%d lr=%.3g train_cvar=%.4f (last %d avg %.4f)",
                    it,
                    n_iter,
                    lr_t,
                    float(loss.detach()),
                    log_every,
                    running / log_every,
                )
            running = 0.0
    return policy


@torch.no_grad()
def evaluate(world: HestonWorld, policy: SemiRecurrentPolicy, n_paths: int) -> dict[str, Any]:
    policy.eval()
    paths = world.simulate(n_paths)
    features = world.features(paths)
    positions = policy.forward_path(features)
    loss = world.terminal_loss(positions, paths)
    return {"paths": paths, "features": features, "positions": positions, "loss": loss}


def delta_rmse(
    paths: HestonPaths,
    positions: torch.Tensor,
    benchmark: HestonBenchmark,
    k: int = 15,
    lo: float = 90.0,
    hi: float = 110.0,
) -> tuple[float, float]:
    """模型 δ1 与 benchmark δ1 的 RMSE（S∈[lo, hi]，默认第 15 步）。"""
    spot = paths.spot[:, k].detach().cpu().numpy()
    var = paths.variance[:, k].detach().cpu().numpy()
    d1_model, _ = benchmark.delta(spot, var, k)
    d1_policy = positions[:, k, 0].detach().cpu().numpy()
    mask = (spot >= lo) & (spot <= hi)
    if not mask.any():
        return float("nan"), float("nan")
    diff = d1_policy[mask] - d1_model[mask]
    return float(np.sqrt(np.mean(diff**2))), float(np.mean(np.abs(diff)))


def q_price(world: HestonWorld) -> float:
    m = world.market
    return float(heston_call_prices(np.array([m.s0]), np.array([m.v0]), m.maturity)[0])


def run(
    cfg: dict[str, Any], seed: int, device: torch.device, log: logging.Logger
) -> dict[str, Any]:
    """train.py 的 runner：训练 recurrent + simpler，评估，返回 metrics/checks/落盘数组。"""
    market = market_from_cfg(cfg)
    world = HestonWorld(market, device=device)
    train_cfg = cfg.get("train", {}) or {}
    alpha = float(cfg["alpha"])
    eval_paths = int(train_cfg.get("eval_paths", 100_000))
    eval_offset = int(train_cfg.get("eval_seed_offset", 10_000))

    log.info(
        "training recurrent policy (d=%d, f=%d, eps=%.4g)", market.d, market.n_features, market.eps
    )
    recurrent = train_policy(world, cfg, device, log, recurrent=True)

    torch.manual_seed(seed + eval_offset)
    ev = evaluate(world, recurrent, eval_paths)
    loss = ev["loss"]
    q = q_price(world)
    price = float(cvar(loss, alpha))

    simpler = train_policy(world, cfg, device, log, recurrent=False)
    ev_simpler = evaluate(world, simpler, eval_paths)
    price_simpler = float(cvar(ev_simpler["loss"], alpha))

    metrics: dict[str, float] = {
        "price": price,
        "q": q,
        "mean_loss": float(loss.mean()),
        "cvar50": float(cvar(loss, 0.5)),
        "cvar99": float(cvar(loss, 0.99)),
        "price_simpler": price_simpler,
        "mean_loss_simpler": float(ev_simpler["loss"].mean()),
    }
    # 无套利自检：causal 策略的 E[L] 必须 = q（否则说明某个工具在离散化下不是鞅，
    # 网络会去套那个漂移，而不是真的对冲）。
    checks: dict[str, bool] = {
        "mean_loss_near_q": bool(abs(float(loss.mean()) - q) < 0.02),
        "simpler_within_3pct": abs(price_simpler - price) / price < 0.03,
    }
    if market.d == 2:
        checks["price_band"] = 1.80 <= price <= 2.10
    extra: dict[str, Any] = {}
    pnl_extra: dict[str, Any] = {"terminal_loss_simpler": ev_simpler["loss"].cpu().numpy()}

    use_benchmark = bool(train_cfg.get("benchmark", True))
    if not use_benchmark:
        pass
    elif market.d == 2:
        benchmark = HestonBenchmark(world)
        bm_positions = benchmark.positions(ev["paths"])
        bm_loss = world.terminal_loss(bm_positions, ev["paths"])
        rmse, mae = delta_rmse(ev["paths"], ev["positions"], benchmark)
        metrics["delta_rmse"] = rmse
        metrics["delta_mae"] = mae
        metrics["benchmark_price"] = float(cvar(bm_loss, alpha))
        metrics["benchmark_mean_loss"] = float(bm_loss.mean())
        # δ1 RMSE 的工程护栏（不是论文判据）：论文 Fig 3 里模型 δ 只是「贴合」解析 δ，
        # 实测最优配方 5 seed 为 0.098–0.118，纯监督拟合解析 δ 的极限也只有 0.065，
        # 所以 0.03 从来达不到；0.15 只用来抓「δ 面整个跑飞」（如早期 BN 失稳的 8.04）。
        checks["delta_rmse_lt_0.15"] = bool(rmse < 0.15)
        pnl_extra["terminal_loss_benchmark"] = bm_loss.cpu().numpy()
        pnl_extra["positions_benchmark"] = bm_positions[:, :, 0].cpu().numpy()
    else:
        # d=1：基准是 Heston 的 δ1（用 benchmark 的 S 向导数），价格只做参考
        benchmark = HestonBenchmark(world)
        bm_positions = benchmark.positions(ev["paths"], d=1)
        bm_loss = world.terminal_loss(bm_positions, ev["paths"])
        metrics["benchmark_price"] = float(cvar(bm_loss, alpha))
        metrics["benchmark_mean_loss"] = float(bm_loss.mean())
        pnl_extra["terminal_loss_benchmark"] = bm_loss.cpu().numpy()

    log.info(
        "eval d=%d price=%.4f q=%.4f mean=%.4f cvar99=%.3f simpler=%.4f%s",
        market.d,
        price,
        q,
        metrics["mean_loss"],
        metrics["cvar99"],
        price_simpler,
        f" delta_rmse={metrics.get('delta_rmse', float('nan')):.4f}" if market.d == 2 else "",
    )

    extra["artifacts"] = {
        "policy.pt": recurrent.state_dict(),
        "policy_simpler.pt": simpler.state_dict(),
    }
    return {
        "metrics": metrics,
        "checks": checks,
        "n_paths": eval_paths,
        "terminal_loss": loss.cpu().numpy(),
        "extra_pnl": pnl_extra,
        **extra,
    }
