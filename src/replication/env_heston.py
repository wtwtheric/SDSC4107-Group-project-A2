"""Heston d=2（标的 + 方差互换）世界：论文 §5.2 的离散时间市场。

- 路径用 pfhedge.stochastic.generate_heston（Andersen QE-M，dt = 1/365，31 个时点）。
- 第二只对冲工具严格按论文 (5.3)–(5.4)：
      S²_k = ∫_0^{t_k} V_s ds + L(t_k, V_k),  L(t,v) = (v−b)/α·(1−e^{−α(T−t)}) + b(T−t)
  它是固定到期日 T 的方差互换的期望总方差（鞅），t=0 时 = bT ≈ 0.0033；
  ∫V ds 每步取条件期望 (v−b)(1−e^{−αΔ})/α + bΔ，故 E[S²_{k+1}|F_k] = S²_k 严格成立
  （左矩形会带来状态相关漂移，见模块内 simulate 的注释）。
- 特征（spec §5.1 冻结）：d=2 为 [log(S_k/S_0), t_k/T, S²_k/S²_0]；d=1 去掉末项。
- 成本按名义本金（论文 (5.7)）：c_k = Σ_i ε |δ_i,k − δ_i,k−1| S_i,k，δ_−1 = 0 计入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from pfhedge.stochastic import generate_heston
from torch import Tensor

from src.common.metrics import proportional_cost
from src.replication.pricing import PAPER, varswap_price_total


@dataclass(frozen=True)
class HestonMarket:
    """市场与实验口径；默认值 = 论文 §5.2。"""

    kappa: float = PAPER.kappa
    theta: float = PAPER.theta
    sigma: float = PAPER.sigma
    rho: float = PAPER.rho
    v0: float = PAPER.v0
    s0: float = PAPER.s0
    strike: float = PAPER.strike
    maturity: float = PAPER.maturity
    n_steps: int = 30
    eps: float = 0.0
    d: int = 2

    @property
    def dt(self) -> float:
        return self.maturity / self.n_steps

    @property
    def n_features(self) -> int:
        return 3 if self.d == 2 else 2

    def time_grid(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        return torch.arange(self.n_steps + 1, dtype=dtype, device=device) * self.dt


class HestonPaths(NamedTuple):
    spot: Tensor  # (B, n+1)
    variance: Tensor  # (B, n+1)
    varswap: Tensor  # (B, n+1)：论文 (5.4) 的 S²（期望总方差，鞅）
    time: Tensor  # (n+1,)


class HestonWorld:
    """离散时间 Heston 世界；d=1 只有标的腿，d=2 加方差互换腿。"""

    def __init__(
        self,
        market: HestonMarket,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.market = market
        self.device = torch.device(device)
        self.dtype = dtype

    # ---------------------------------------------------------------- 模拟
    def simulate(self, n_paths: int) -> HestonPaths:
        """采样 n_paths 条路径；随机数走全局 torch RNG（seed_everything 已设好）。"""
        m = self.market
        spot, variance = generate_heston(
            n_paths=n_paths,
            n_steps=m.n_steps + 1,
            init_state=(m.s0, m.v0),
            kappa=m.kappa,
            theta=m.theta,
            sigma=m.sigma,
            rho=m.rho,
            dt=m.dt,
            dtype=self.dtype,
            device=self.device,
        )
        time = m.time_grid(self.dtype, self.device)
        var_np = variance.cpu().numpy()
        t_np = time.cpu().numpy()
        # ∫_0^{t_k} V ds：每步用条件期望 ∫_Δ V ds = (v−b)(1−e^{−αΔ})/α + bΔ
        # （CIR 精确转移的一阶矩；用左矩形会引入 −(V_j−b)αΔ²/2 的状态相关偏差，
        #  使 S² 不再是鞅，网络可借机套利 —— 见 docs/b0.md 的鞅检查。）
        decay = np.expm1(-m.kappa * m.dt)  # e^{−αΔ} − 1 < 0
        accrual = (var_np[:, :-1] - m.theta) * (-decay / m.kappa) + m.theta * m.dt
        integrated = np.zeros_like(var_np)
        integrated[:, 1:] = np.cumsum(accrual, axis=1)
        varswap = integrated + varswap_price_total(var_np, t_np[None, :], m.maturity)
        return HestonPaths(
            spot=spot,
            variance=variance,
            varswap=torch.as_tensor(varswap, dtype=self.dtype, device=self.device),
            time=time,
        )

    # ---------------------------------------------------------------- 特征
    def features(self, paths: HestonPaths) -> Tensor:
        """(B, n, f)：只到 k = n−1（k = n 不交易）。"""
        m = self.market
        n = m.n_steps
        log_ret = torch.log(paths.spot[:, :n] / m.s0)
        t_norm = (paths.time[:n] / m.maturity).unsqueeze(0).expand(n_paths(paths.spot), n)
        if m.d == 1:
            return torch.stack([log_ret, t_norm], dim=-1)
        ratio = paths.varswap[:, :n] / paths.varswap[:, :1]
        return torch.stack([log_ret, t_norm, ratio], dim=-1)

    # ---------------------------------------------------------------- 价格 / 损益
    def prices(self, paths: HestonPaths) -> Tensor:
        """(B, n+1, d)：对冲工具价格（S¹_k, S²_k），k = 0..n。"""
        if self.market.d == 1:
            return paths.spot.unsqueeze(-1)
        return torch.stack([paths.spot, paths.varswap], dim=-1)

    def payoff(self, paths: HestonPaths) -> Tensor:
        return torch.clamp(paths.spot[:, -1] - self.market.strike, min=0.0)

    def cost(self, positions: Tensor, paths: HestonPaths) -> Tensor:
        prices = self.prices(paths)[..., : positions.shape[-1]]
        return proportional_cost(positions, prices, self.market.eps)

    def terminal_loss(self, positions: Tensor, paths: HestonPaths) -> Tensor:
        """L = Z − Σ_k δ_k·(S_{k+1} − S_k) + 成本（positions (B, n, d)）。"""
        prices = self.prices(paths)[..., : positions.shape[-1]]
        step_pnl = (positions * (prices[:, 1:] - prices[:, :-1])).sum(dim=-1)
        hedge = step_pnl.sum(dim=-1)
        return self.payoff(paths) - hedge + self.cost(positions, paths)

    def realized_variance(self, paths: HestonPaths) -> Tensor:
        """(B,)：全区间年化已实现方差 365·Σ (Δlog S)² / (n·Δ) → 用于 B0 对拍。"""
        log_ret = torch.log(paths.spot[:, 1:] / paths.spot[:, :-1])
        return log_ret.pow(2).sum(dim=-1) / self.market.maturity


def n_paths(spot: Tensor) -> int:
    return int(spot.shape[0])
