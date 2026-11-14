"""随机性与设备（spec §8：所有随机源从这里取）。"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
    """同时设 python random / numpy / torch 的种子，返回 seed 本身。"""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def resolve_device(spec: str | torch.device | None = "auto") -> torch.device:
    """auto -> cuda 可用就用 cuda，否则 cpu；显式 cuda 但不可用时直接报错。"""
    if isinstance(spec, torch.device):
        return spec
    if spec in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(str(spec))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device={spec!r} 要求 CUDA，但 torch.cuda.is_available() 为 False")
    return device


def torch_info(device: torch.device | None = None) -> dict[str, Any]:
    """写进 metrics.json 的 provenance：torch 版本 + 实际设备名。"""
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if device is not None and device.type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(device)
    else:
        info["device_name"] = "cpu"
    return info
