"""Heston d=2 (stock + variance swap) environment used by A1/A2 experiments.

Two simulation/feature modes are supported so paper-fidelity effects can be
isolated without changing the validated baseline:

- simulator="qem" (default): PFHedge Andersen QE-M paths, with the existing
  conditional-mean variance-integral construction used by the baseline.
- simulator="paper_sbk": exact CIR endpoint transitions (non-central chi-square)
  followed by the paper's simplified Broadie-Kaya discretization: integrated
  variance is replaced by a left-point sum and the stock transition uses the
  Heston conditional log-price identity with that approximate integral.

Feature sets:
- feature_set="current" (default): [log(S/S0), t/T, S2/S2_0].
- feature_set="paper": [log(S), V], matching Section 5.2.2 of the paper.

Existing configs omit both fields and therefore remain byte-for-byte equivalent
in behaviour to the original QE-M/current-input baseline.
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
    """Market and experiment convention; numerical defaults match paper Section 5.2."""

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
    simulator: str = "qem"
    feature_set: str = "current"

    @property
    def dt(self) -> float:
        return self.maturity / self.n_steps

    @property
    def n_features(self) -> int:
        if self.d == 2 and self.feature_set == "paper":
            return 2
        return 3 if self.d == 2 else 2

    def time_grid(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        return torch.arange(self.n_steps + 1, dtype=dtype, device=device) * self.dt


class HestonPaths(NamedTuple):
    spot: Tensor
    variance: Tensor
    varswap: Tensor
    time: Tensor


def _sample_exact_cir_transition(
    variance: Tensor,
    *,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
) -> Tensor:
    """Sample the exact CIR endpoint transition via the Poisson-mixture identity.

    If X ~ noncentral-chi-square(df, lambda), then conditional on
    N ~ Poisson(lambda/2), X is central chi-square with df+2N degrees
    of freedom. A central chi-square is Gamma(shape=df/2+N, rate=1/2).
    """
    decay = float(np.exp(-kappa * dt))
    one_minus_decay = 1.0 - decay
    scale = sigma * sigma * one_minus_decay / (4.0 * kappa)
    df = 4.0 * kappa * theta / (sigma * sigma)
    noncentrality = (4.0 * kappa * decay * variance / (sigma * sigma * one_minus_decay)).clamp_min(
        0.0
    )
    poisson = torch.poisson(noncentrality / 2.0)
    concentration = poisson + df / 2.0
    rate = torch.full_like(concentration, 0.5)
    chi_square = torch.distributions.Gamma(concentration, rate).sample()
    return scale * chi_square


def _generate_paper_sbk(
    market: HestonMarket,
    *,
    n_paths: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate paper-style simplified Broadie-Kaya Heston paths.

    The paper states that it samples the CIR transition exactly and then uses
    a simplified Broadie-Kaya scheme, corresponding to replacing V in the
    stock SDE by a piecewise-constant process and the variance integral by a
    sum. We implement precisely that stated approximation on the daily grid.

    For I_k = V_k * dt, the Heston identity gives
      d log S = -I/2 + rho/sigma * (dV-kappa*theta*dt+kappa*I)
                + sqrt(1-rho^2)*sqrt(I)*Z.
    """
    m = market
    spot = torch.empty((n_paths, m.n_steps + 1), dtype=dtype, device=device)
    variance = torch.empty_like(spot)
    integrated = torch.zeros_like(spot)
    spot[:, 0] = m.s0
    variance[:, 0] = m.v0
    rho_orth = float(np.sqrt(max(0.0, 1.0 - m.rho * m.rho)))

    for k in range(m.n_steps):
        v_now = variance[:, k]
        v_next = _sample_exact_cir_transition(
            v_now,
            kappa=m.kappa,
            theta=m.theta,
            sigma=m.sigma,
            dt=m.dt,
        )
        integrated_step = v_now.clamp_min(0.0) * m.dt
        normal = torch.randn(n_paths, dtype=dtype, device=device)
        correlated_term = (m.rho / m.sigma) * (
            v_next - v_now - m.kappa * m.theta * m.dt + m.kappa * integrated_step
        )
        orthogonal_term = rho_orth * torch.sqrt(integrated_step) * normal
        log_return = -0.5 * integrated_step + correlated_term + orthogonal_term
        spot[:, k + 1] = spot[:, k] * torch.exp(log_return)
        variance[:, k + 1] = v_next
        integrated[:, k + 1] = integrated[:, k] + integrated_step

    return spot, variance, integrated


class HestonWorld:
    """Discrete-time Heston world; d=2 adds the variance-swap hedge leg."""

    def __init__(
        self,
        market: HestonMarket,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.market = market
        self.device = torch.device(device)
        self.dtype = dtype
        if market.simulator not in {"qem", "paper_sbk"}:
            raise ValueError(f"unknown Heston simulator={market.simulator!r}")
        if market.feature_set not in {"current", "paper"}:
            raise ValueError(f"unknown Heston feature_set={market.feature_set!r}")
        if market.feature_set == "paper" and market.d != 2:
            raise ValueError("paper feature set [log(S), V] is defined here only for d=2")

    def simulate(self, n_paths: int) -> HestonPaths:
        """Sample paths using the configured simulator and global torch RNG."""
        m = self.market
        time = m.time_grid(self.dtype, self.device)

        if m.simulator == "qem":
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
            var_np = variance.cpu().numpy()
            t_np = time.cpu().numpy()
            decay = np.expm1(-m.kappa * m.dt)
            accrual = (var_np[:, :-1] - m.theta) * (-decay / m.kappa) + m.theta * m.dt
            integrated = np.zeros_like(var_np)
            integrated[:, 1:] = np.cumsum(accrual, axis=1)
            varswap = integrated + varswap_price_total(var_np, t_np[None, :], m.maturity)
            varswap_tensor = torch.as_tensor(varswap, dtype=self.dtype, device=self.device)
        else:
            spot, variance, integrated_tensor = _generate_paper_sbk(
                m,
                n_paths=n_paths,
                dtype=self.dtype,
                device=self.device,
            )
            tau = (m.maturity - time).clamp_min(0.0)
            continuation = (variance - m.theta) / m.kappa * (
                1.0 - torch.exp(-m.kappa * tau.unsqueeze(0))
            ) + m.theta * tau.unsqueeze(0)
            varswap_tensor = integrated_tensor + continuation

        return HestonPaths(
            spot=spot,
            variance=variance,
            varswap=varswap_tensor,
            time=time,
        )

    def features(self, paths: HestonPaths) -> Tensor:
        """Return the configured per-date information set for dates 0..n-1."""
        m = self.market
        n = m.n_steps
        if m.feature_set == "paper":
            return torch.stack(
                [
                    torch.log(paths.spot[:, :n]),
                    paths.variance[:, :n],
                ],
                dim=-1,
            )

        log_ret = torch.log(paths.spot[:, :n] / m.s0)
        t_norm = (paths.time[:n] / m.maturity).unsqueeze(0).expand(n_paths(paths.spot), n)
        if m.d == 1:
            return torch.stack([log_ret, t_norm], dim=-1)
        ratio = paths.varswap[:, :n] / paths.varswap[:, :1]
        return torch.stack([log_ret, t_norm, ratio], dim=-1)

    def prices(self, paths: HestonPaths) -> Tensor:
        if self.market.d == 1:
            return paths.spot.unsqueeze(-1)
        return torch.stack([paths.spot, paths.varswap], dim=-1)

    def payoff(self, paths: HestonPaths) -> Tensor:
        return torch.clamp(paths.spot[:, -1] - self.market.strike, min=0.0)

    def cost(self, positions: Tensor, paths: HestonPaths) -> Tensor:
        prices = self.prices(paths)[..., : positions.shape[-1]]
        return proportional_cost(positions, prices, self.market.eps)

    def terminal_loss(self, positions: Tensor, paths: HestonPaths) -> Tensor:
        prices = self.prices(paths)[..., : positions.shape[-1]]
        step_pnl = (positions * (prices[:, 1:] - prices[:, :-1])).sum(dim=-1)
        hedge = step_pnl.sum(dim=-1)
        return self.payoff(paths) - hedge + self.cost(positions, paths)

    def realized_variance(self, paths: HestonPaths) -> Tensor:
        log_ret = torch.log(paths.spot[:, 1:] / paths.spot[:, :-1])
        return log_ret.pow(2).sum(dim=-1) / self.market.maturity


def n_paths(spot: Tensor) -> int:
    return int(spot.shape[0])
