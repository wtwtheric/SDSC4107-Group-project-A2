"""Black–Scholes d=1 世界：给 A2 的 E4-BS 用（spec §6 A2 前置）。

与 HestonWorld 同一套 step / payoff / cost 接口；特征 f=2：[log(S_k/S_0), t_k/T]。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

from src.common.metrics import proportional_cost
from src.replication.pricing import bs_call, bs_delta


@dataclass(frozen=True)
class BSMarket:
    sigma: float = 0.2
    s0: float = 100.0
    strike: float = 100.0
    maturity: float = 30.0 / 365.0
    n_steps: int = 30
    eps: float = 0.0
    d: int = 1

    @property
    def dt(self) -> float:
        return self.maturity / self.n_steps

    @property
    def n_features(self) -> int:
        return 2

    def time_grid(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        return torch.arange(self.n_steps + 1, dtype=dtype, device=device) * self.dt


class BSPaths(NamedTuple):
    spot: Tensor  # (B, n+1)
    time: Tensor  # (n+1,)


class BSWorld:
    def __init__(
        self,
        market: BSMarket,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.market = market
        self.device = torch.device(device)
        self.dtype = dtype

    def simulate(self, n_paths: int) -> BSPaths:
        m = self.market
        drift = -0.5 * m.sigma**2 * m.dt
        shock = m.sigma * (m.dt**0.5)
        normals = torch.randn(n_paths, m.n_steps, dtype=self.dtype, device=self.device)
        log_inc = drift + shock * normals
        log_path = torch.cat(
            [torch.zeros(n_paths, 1, dtype=self.dtype, device=self.device), log_inc.cumsum(1)],
            dim=1,
        )
        return BSPaths(spot=m.s0 * log_path.exp(), time=m.time_grid(self.dtype, self.device))

    def features(self, paths: BSPaths) -> Tensor:
        m = self.market
        n = m.n_steps
        log_ret = torch.log(paths.spot[:, :n] / m.s0)
        t_norm = (paths.time[:n] / m.maturity).unsqueeze(0).expand(int(paths.spot.shape[0]), n)
        return torch.stack([log_ret, t_norm], dim=-1)

    def prices(self, paths: BSPaths) -> Tensor:
        return paths.spot.unsqueeze(-1)

    def payoff(self, paths: BSPaths) -> Tensor:
        return torch.clamp(paths.spot[:, -1] - self.market.strike, min=0.0)

    def cost(self, positions: Tensor, paths: BSPaths) -> Tensor:
        return proportional_cost(positions, self.prices(paths), self.market.eps)

    def terminal_loss(self, positions: Tensor, paths: BSPaths) -> Tensor:
        prices = self.prices(paths)
        hedge = (positions * (prices[:, 1:] - prices[:, :-1])).sum(dim=-1).sum(dim=-1)
        return self.payoff(paths) - hedge + self.cost(positions, paths)

    def analytic_price(self, maturity: float | None = None, spot: float | None = None) -> float:
        m = self.market
        return bs_call(spot or m.s0, m.strike, maturity or m.maturity, m.sigma)

    def analytic_delta(self, spot: float, maturity: float) -> float:
        m = self.market
        return bs_delta(spot, m.strike, maturity, m.sigma)
