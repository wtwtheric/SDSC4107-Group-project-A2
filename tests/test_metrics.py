from __future__ import annotations

import numpy as np
import pytest
import torch
from src.common.metrics import (
    capital_requirement,
    cvar,
    entropic,
    indifference_price,
    proportional_cost,
    save_pnl,
    turnover,
)


def test_cvar_hand_calc() -> None:
    x = torch.arange(10, dtype=torch.float64)
    assert float(cvar(x, 0.9)) == pytest.approx(9.0)
    assert float(cvar(x, 0.5)) == pytest.approx(float(np.mean([9, 8, 7, 6, 5])))


def test_cvar_integer_tail_mass() -> None:
    x = torch.arange(10, dtype=torch.float64)
    # (1 - 0.8) * 10 = 2 exactly: mean of the two worst losses, 9 and 8.
    assert float(cvar(x, 0.8)) == pytest.approx(8.5)


def test_cvar_fractional_tail_mass() -> None:
    x = torch.arange(10, dtype=torch.float64)
    # (1 - 0.85) * 10 = 1.5: 9 has full weight, 8 has weight 0.5.
    expected = (9.0 + 0.5 * 8.0) / 1.5
    assert float(cvar(x, 0.85)) == pytest.approx(expected)


def test_cvar_fractional_gradient_weights() -> None:
    x = torch.arange(10, dtype=torch.float64, requires_grad=True)
    value = cvar(x, 0.85)
    value.backward()

    expected_grad = torch.zeros_like(x)
    expected_grad[9] = 1.0 / 1.5
    expected_grad[8] = 0.5 / 1.5
    torch.testing.assert_close(x.grad, expected_grad)


def test_cvar_translation_invariance() -> None:
    x = torch.tensor([-2.0, -0.5, 0.2, 1.7, 3.5, 8.0, 13.0], dtype=torch.float64)
    shift = 4.25
    assert float(cvar(x + shift, 0.83)) == pytest.approx(float(cvar(x, 0.83)) + shift)


def test_capital_requirement() -> None:
    x = torch.arange(100, dtype=torch.float64)
    req = capital_requirement(x)
    assert req["c95"] == pytest.approx(float(np.mean(np.arange(95, 100))))
    assert req["c99"] == pytest.approx(99.0)


def test_turnover_and_cost_hand_calc() -> None:
    positions = torch.tensor([[[1.0], [2.0]]], dtype=torch.float64)  # (1, 2, 1)
    prices = torch.tensor([[[100.0], [101.0], [102.0]]], dtype=torch.float64)
    step = turnover(positions)
    assert step.shape == (1, 2)
    assert float(step.sum()) == pytest.approx(2.0)  # |1-0| + |2-1|
    cost = proportional_cost(positions, prices, 0.01)
    assert float(cost) == pytest.approx(0.01 * (1.0 * 100.0 + 1.0 * 101.0))


def test_proportional_cost_shape_guard() -> None:
    with pytest.raises(ValueError):
        proportional_cost(torch.zeros(2, 3, 1), torch.zeros(2, 2, 1), 0.0)


def test_indifference_price_identity() -> None:
    loss = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    expected = 1.69 + float(cvar(loss - 1.69, 0.5))
    assert indifference_price(1.69, loss, 0.5) == pytest.approx(expected)


def test_entropic_matches_formula() -> None:
    loss = torch.tensor([0.0, 1.0], dtype=torch.float64)
    expected = float(torch.log(torch.exp(loss).mean()))
    assert float(entropic(loss, 1.0)) == pytest.approx(expected)
    assert float(entropic(loss, 0.5)) == pytest.approx(
        float((torch.logsumexp(0.5 * loss, dim=0) - float(np.log(2.0))) / 0.5)
    )


def test_save_pnl_keys(tmp_path) -> None:
    path = save_pnl(tmp_path / "pnl.npz", torch.zeros(4))
    with np.load(path) as data:
        assert set(data.files) == {"terminal_loss"}
        assert data["terminal_loss"].shape == (4,)
