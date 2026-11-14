from __future__ import annotations

import torch
from src.policy.policy import ConstantPolicy, Policy


def test_constant_policy_shapes() -> None:
    policy = ConstantPolicy(d=2, value=0.5)
    x_k = torch.zeros(4, 3)
    out = policy.forward(x_k)
    assert out.shape == (4, 2)
    assert torch.all(out == 0.5)
    policy.reset()


def test_policy_protocol_is_satisfied() -> None:
    assert isinstance(ConstantPolicy(), Policy)
