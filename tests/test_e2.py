"""E2 runner tests: shared-path evaluation and metric semantics, without long training."""

from __future__ import annotations

import logging

import pytest
import torch
from src.replication import e2


def _cfg() -> dict:
    return {
        "exp": "e2",
        "kind": "e2",
        "n_steps": 3,
        "batch": 32,
        "eps": 0.0,
        "alpha": 0.99,
        "compare_alpha": 0.50,
        "policy": {
            "hidden": 17,
            "recurrent": True,
            "norm": "layer",
            "out_scale": [1.0, 10.0],
        },
        "market": {
            "model": "heston",
            "d": 2,
            "s0": 100.0,
            "k": 100.0,
            "t": 3.0 / 365.0,
            "kappa": 1.0,
            "theta": 0.04,
            "sigma": 2.0,
            "rho": -0.7,
            "v0": 0.04,
        },
        "train": {
            "n_iter": 2,
            "lr": 0.001,
            "lr_schedule": "constant",
            "eval_paths": 128,
            "eval_seed_offset": 10_000,
            "log_every": 1,
        },
    }


def test_e2_runner_returns_required_schema_and_positions() -> None:
    payload = e2.run(
        _cfg(),
        seed=0,
        device=torch.device("cpu"),
        log=logging.getLogger("test.e2"),
    )

    metrics = payload["metrics"]
    required = {
        "price",
        "cvar05_own",
        "cvar99_own",
        "cvar05_other",
        "cvar99_other",
        "mean_abs_hedging_error_alpha50",
        "mean_abs_hedging_error_alpha99",
    }
    assert required <= set(metrics)
    assert all(torch.isfinite(torch.tensor(metrics[key])) for key in required)
    assert payload["terminal_loss"].shape == (128,)
    assert set(payload["artifacts"]) == {"policy_alpha50.pt", "policy_alpha99.pt"}

    extra = payload["extra_pnl"]
    for key in (
        "terminal_loss_alpha50",
        "terminal_loss_alpha99",
        "loss_above_q_alpha50",
        "loss_above_q_alpha99",
        "hedging_error_common_q_alpha50",
        "hedging_error_common_q_alpha99",
        "hedging_error_own_price_alpha50",
        "hedging_error_own_price_alpha99",
    ):
        assert extra[key].shape == (128,)
    assert extra["positions_alpha50"].shape == (128, 3, 2)
    assert extra["positions_alpha99"].shape == (128, 3, 2)
    assert extra["positions_alpha50"].dtype == "float32"


def test_e2_common_premium_translation_identity() -> None:
    payload = e2.run(
        _cfg(),
        seed=1,
        device=torch.device("cpu"),
        log=logging.getLogger("test.e2.identity"),
    )
    metrics = payload["metrics"]

    # CVaR is cash-translation invariant: p_alpha = q + CVaR_alpha(L-q).
    assert metrics["price_alpha99"] == pytest.approx(
        metrics["q"] + metrics["cvar99_own"], abs=1e-10
    )
    assert metrics["price_alpha50"] == pytest.approx(
        metrics["q"] + metrics["cvar05_other"], abs=1e-10
    )

    extra = payload["extra_pnl"]
    torch.testing.assert_close(
        torch.as_tensor(extra["hedging_error_common_q_alpha50"]),
        -torch.as_tensor(extra["loss_above_q_alpha50"]),
    )
    torch.testing.assert_close(
        torch.as_tensor(extra["hedging_error_own_price_alpha99"]),
        metrics["price_alpha99"] - torch.as_tensor(extra["terminal_loss_alpha99"]),
    )
