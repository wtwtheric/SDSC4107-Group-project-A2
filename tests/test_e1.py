"""E1 runner 的轻量单测：接口与落盘字段（不跑真训练）。"""

from __future__ import annotations

import logging

import torch
from src.replication import e1


def _cfg(**over: object) -> dict:
    cfg = {
        "exp": "e1",
        "kind": "e1",
        "n_steps": 5,
        "batch": 32,
        "eps": 0.0,
        "alpha": 0.5,
        "policy": {"hidden": 16, "recurrent": True, "batch_norm": True},
        "market": {"d": 1, "s0": 100.0, "k": 100.0, "t": 5 / 365},
        "train": {"n_iter": 3, "eval_paths": 256, "log_every": 1, "benchmark": False},
    }
    cfg.update(over)
    return cfg


def test_runner_returns_metrics_and_artifacts(caplog) -> None:
    caplog.set_level(logging.INFO)
    payload = e1.run(_cfg(), seed=0, device=torch.device("cpu"), log=logging.getLogger("test"))
    metrics = payload["metrics"]
    for key in ("price", "q", "mean_loss", "cvar50", "cvar99", "price_simpler"):
        assert key in metrics
        assert torch.isfinite(torch.tensor(metrics[key]))
    assert payload["terminal_loss"].shape == (256,)
    assert set(payload["artifacts"]) == {"policy.pt", "policy_simpler.pt"}
    # price_band（1.80–2.10）是 E1 d=2 的论文靶子；d=1 的 e1d1 没有这个判据
    assert "price_band" not in payload["checks"]


def test_runner_d2_without_benchmark_has_no_rmse() -> None:
    cfg = _cfg(market={"d": 2, "s0": 100.0, "k": 100.0, "t": 5 / 365})
    payload = e1.run(cfg, seed=1, device=torch.device("cpu"), log=logging.getLogger("test"))
    assert "delta_rmse" not in payload["metrics"]
    assert "price_band" in payload["checks"]
    assert payload["terminal_loss"].shape == (256,)
