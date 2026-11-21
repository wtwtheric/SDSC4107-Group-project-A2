"""Small CPU smoke tests for entropic-risk training in both market worlds."""
from __future__ import annotations

import logging

import torch
from src.replication.e4 import run_one


def _cfg(exp: str) -> dict:
    market = {"s0": 100.0, "k": 100.0, "t": 3 / 365}
    if exp == "e4bs":
        market.update({"model": "bs", "d": 1, "sigma": 0.2})
        scale = 1.0
        hidden = 8
    else:
        market.update({"model": "heston", "d": 2, "kappa": 1.0, "theta": 0.04,
                       "sigma": 2.0, "rho": -0.7, "v0": 0.04})
        scale = [1.0, 10.0]
        hidden = 8
    return {
        "exp": exp, "n_steps": 3, "batch": 16, "eps": 0.01,
        "policy": {"hidden": hidden, "norm": "layer", "out_scale": scale},
        "market": market, "risk": {"lambda": 1.0},
        "train": {"n_iter": 1, "lr": 0.001, "lr_schedule": "constant", "log_every": 1,
                  "lr_min": 0.0005, "eval_paths": 64, "eval_batch_size": 16, "eval_seed": 123},
    }


def test_e4_bs_runner_returns_entropic_price_increment() -> None:
    result = run_one(_cfg("e4bs"), 0, torch.device("cpu"), logging.getLogger("test.e4bs"), 0.01)
    assert result["n_paths"] == 64
    assert result["terminal_loss"].shape == (64,)
    assert result["metrics"]["price_eps"] > 0.0
    assert result["metrics"]["price0"] > 0.0
    assert "policy.pt" in result["artifacts"]


def test_e4_heston_runner_smoke() -> None:
    result = run_one(_cfg("e4h"), 1, torch.device("cpu"), logging.getLogger("test.e4h"), 0.01)
    assert result["n_paths"] == 64
    assert result["terminal_loss"].shape == (64,)
    assert result["metrics"]["price0"] > 0.0
