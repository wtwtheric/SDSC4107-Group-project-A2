"""Small CPU smoke test for the E3 runner; no long training."""

from __future__ import annotations

import logging

import pytest
import torch
from src.replication.e3 import run


def test_e3_runner_returns_cost_breakdown_and_paired_losses(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    cfg = {
        "exp": "e3",
        "n_steps": 3,
        "batch": 16,
        "eps": 0.01,
        "alpha": 0.99,
        "policy": {"hidden": 8, "norm": "layer", "out_scale": [1.0, 10.0]},
        "market": {
            "d": 2,
            "s0": 100.0,
            "k": 100.0,
            "t": 3 / 365,
            "kappa": 1.0,
            "theta": 0.04,
            "sigma": 2.0,
            "rho": -0.7,
            "v0": 0.04,
        },
        "train": {
            "n_iter": 1,
            "lr": 0.001,
            "lr_schedule": "constant",
            "log_every": 1,
            "eval_paths": 64,
            "eval_batch_size": 16,
            "eval_seed": 123,
            "visual_paths": 32,
            "visual_step": 1,
            "validation_paths": 16,
            "validation_batch_size": 8,
            "validation_seed": 777,
        },
    }
    result = run(cfg, 0, torch.device("cpu"), logging.getLogger("test.e3"))
    assert result["n_paths"] == 64
    assert result["terminal_loss"].shape == (64,)
    assert result["extra_pnl"]["terminal_loss_simpler"].shape == (64,)
    assert result["metrics"]["mean_cost_recurrent"] >= 0.0
    assert result["metrics"]["mean_stock_cost_recurrent"] >= 0.0
    assert result["metrics"]["mean_varswap_cost_recurrent"] >= 0.0
    assert result["metrics"]["mean_cost_recurrent"] == pytest.approx(
        result["metrics"]["mean_stock_cost_recurrent"]
        + result["metrics"]["mean_varswap_cost_recurrent"],
        abs=1e-10,
    )
    assert set(result["artifacts"]) == {"policy_recurrent.pt", "policy_simpler.pt"}
    assert "validation_cvar=" in caplog.text
