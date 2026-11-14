from __future__ import annotations

import numpy as np
import torch
from src.common.seed import resolve_device, seed_everything


def test_same_seed_same_streams() -> None:
    seed_everything(3)
    first = torch.rand(3)
    first_np = np.random.rand(3)
    seed_everything(3)
    assert torch.equal(first, torch.rand(3))
    assert np.allclose(first_np, np.random.rand(3))


def test_different_seed_differs() -> None:
    seed_everything(1)
    a = torch.rand(3)
    seed_everything(2)
    assert not torch.equal(a, torch.rand(3))


def test_resolve_device() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda"}
