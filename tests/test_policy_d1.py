"""A1 → B2 交接件 policy_heston_d1 的冒烟测试（权重由 scripts/export_policy_d1.py 产出）。

没有导出过权重时跳过：测试不负责训练，只保证"拿到文件就能跑"。
"""

from __future__ import annotations

import pytest
import torch
from src.policy.heston_d1 import WEIGHTS, load, read_checkpoint


@pytest.mark.skipif(not WEIGHTS.exists(), reason="run scripts/export_policy_d1.py first")
def test_checkpoint_meta_matches_loader() -> None:
    payload = read_checkpoint()
    meta = payload["meta"]
    assert meta["d"] == 1
    assert meta["n_features"] == 2
    assert meta["n_steps"] == 30
    assert meta["dtype"] == "float64"
    assert "state_dict" in payload


@pytest.mark.skipif(not WEIGHTS.exists(), reason="run scripts/export_policy_d1.py first")
def test_loaded_policy_returns_finite_positions() -> None:
    policy = load()
    policy.reset()
    x = torch.tensor([[0.0, 0.0], [-0.1, 0.5], [0.1, 1.0]], dtype=torch.float64)
    delta = policy(x)
    assert delta.shape == (3, 1)
    assert torch.isfinite(delta).all()
    # ATM 的标的仓位应在 [0, 1] 附近，量级不对说明 out_scale/norm 装配错了
    assert 0.2 < float(delta[0, 0]) < 1.2
