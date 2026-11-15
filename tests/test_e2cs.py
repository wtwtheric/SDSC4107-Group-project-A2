"""E2 call-spread payoff and runner contract tests."""
from __future__ import annotations

import pytest
import torch
from src.replication.e2cs import CallSpreadWorld, _model_delta_grids
from src.replication.env_heston import HestonMarket, HestonPaths


def _paths(terminal: list[float]) -> HestonPaths:
    len(terminal)# n = len(terminal)
    spot = torch.tensor([[100.0, x] for x in terminal], dtype=torch.float64)
    variance = torch.full_like(spot, 0.04)
    varswap = torch.full_like(spot, 0.04 * (30.0 / 365.0))
    time = torch.tensor([0.0, 30.0 / 365.0], dtype=torch.float64)
    return HestonPaths(spot, variance, varswap, time)


def test_normalized_call_spread_payoff_values() -> None:
    world = CallSpreadWorld(HestonMarket(n_steps=1), 100.0, 101.0)
    payoff = world.payoff(_paths([99.0, 100.0, 100.5, 101.0, 102.0]))
    torch.testing.assert_close(payoff, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0], dtype=torch.float64))


def test_call_spread_rejects_invalid_strikes() -> None:
    with pytest.raises(ValueError, match="strike2 > strike1"):
        CallSpreadWorld(HestonMarket(n_steps=1), 101.0, 100.0)


def test_call_spread_payoff_is_bounded_and_batch_shaped() -> None:
    world = CallSpreadWorld(HestonMarket(n_steps=1), 100.0, 101.0)
    payoff = world.payoff(_paths([0.0, 99.0, 100.5, 101.0, 200.0]))
    assert payoff.shape == (5,)
    assert bool(torch.all((payoff >= 0.0) & (payoff <= 1.0)))


def test_model_call_spread_delta_grids_are_finite() -> None:
    market = HestonMarket(n_steps=30)
    stock, varswap = _model_delta_grids(
        market,
        100.0,
        101.0,
        15,
        torch.tensor([90.0, 100.0, 110.0]).numpy(),
        torch.tensor([0.02, 0.04, 0.08]).numpy(),
    )
    assert stock.shape == (3, 3)
    assert varswap.shape == (3, 3)
    assert bool(torch.isfinite(torch.as_tensor(stock)).all())
    assert bool(torch.isfinite(torch.as_tensor(varswap)).all())
