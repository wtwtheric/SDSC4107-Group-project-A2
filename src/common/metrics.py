"""共享指标与结果落盘（spec §5.2 / §5.3 / §5.4；A1 维护，A2/B2 复用）。"""

from __future__ import annotations

import datetime as dt
import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.common.config import REPO_ROOT, digest_config
from src.common.types import Tensor


def cvar(x: Tensor, alpha: float) -> Tensor:
    """平均 VaR / CVaR：最差 (1-alpha) 比例的均值；alpha 是置信水平（0.5 / 0.95 / 0.99）。"""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not isinstance(x, torch.Tensor):  # 便利：也接受 ndarray / list
        x = torch.as_tensor(np.asarray(x, dtype=float))
    flat = x.reshape(-1)
    n = flat.numel()
    if n == 0:
        raise ValueError("cvar() got an empty tensor")
    k = max(1, math.ceil((1.0 - alpha) * n - 1e-12))
    return torch.topk(flat, k, largest=True).values.mean()


def capital_requirement(loss: Tensor) -> dict[str, float]:
    """C* = CVaR(loss)，闭式（spec §5.4），返回 {"c95": ..., "c99": ...}。"""
    return {"c95": float(cvar(loss, 0.95)), "c99": float(cvar(loss, 0.99))}


def turnover(positions: Tensor) -> Tensor:
    """每步换手 Σ_i |δ_i,t − δ_i,t−1|（δ_-1 = 0）；输入 (..., n, d)，返回 (..., n)。"""
    if positions.ndim < 2:
        raise ValueError(f"positions must be (..., n, d), got {tuple(positions.shape)}")
    prev = torch.cat([torch.zeros_like(positions[..., :1, :]), positions[..., :-1, :]], dim=-2)
    return (positions - prev).abs().sum(dim=-1)


def proportional_cost(positions: Tensor, prices: Tensor, eps: Tensor | float) -> Tensor:
    """比例成本 c = Σ_t Σ_i eps_i |δ_i,t − δ_i,t−1| S_i,t（论文 (5.7)）。

    positions (..., n, d)，prices (..., n+1, d)，eps 标量或 (d,)；返回 (...,)。
    """
    if positions.ndim != prices.ndim:
        raise ValueError("positions/prices must share rank")
    if positions.shape[-1] != prices.shape[-1] or prices.shape[-2] != positions.shape[-2] + 1:
        raise ValueError(
            "expect positions (..., n, d) and prices (..., n+1, d); got "
            f"{tuple(positions.shape)} / {tuple(prices.shape)}"
        )
    prev = torch.cat([torch.zeros_like(positions[..., :1, :]), positions[..., :-1, :]], dim=-2)
    trades = (positions - prev).abs() * prices[..., : positions.shape[-2], :]
    coef = torch.as_tensor(eps, dtype=prices.dtype, device=prices.device)
    return (trades * coef).sum(dim=(-1, -2))


def indifference_price(q: float, loss: Tensor, alpha: float) -> float:
    """无差异价格 p0（解 rho(L − p0) = 0）：p0 = q + CVaR_alpha(L − q)。"""
    return float(q + cvar(loss - q, alpha))


def entropic(loss: Tensor, lam: float = 1.0) -> Tensor:
    """熵风险测度 rho(L) = (1/lam) log E[exp(lam L)]，用 logsumexp 防溢出。"""
    if lam <= 0:
        raise ValueError(f"lam must be > 0, got {lam}")
    flat = loss.reshape(-1)
    return (torch.logsumexp(lam * flat, dim=0) - math.log(flat.numel())) / lam


def git_provenance(root: Path = REPO_ROOT) -> dict[str, Any]:
    """git_commit + dirty（spec §4：脏工作区的数字不许进报告）。"""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "dirty": True}
    return {"git_commit": commit, "dirty": bool(status.strip())}


def save_pnl(path: str | Path, terminal_loss: Tensor | np.ndarray, **extra: Any) -> Path:
    """按 spec §5.3 落 pnl.npz；terminal_loss 必填，其余键可选（wealth_path / positions）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"terminal_loss": np.asarray(terminal_loss)}
    arrays.update({key: np.asarray(value) for key, value in extra.items()})
    np.savez_compressed(target, **arrays)
    return target


def write_metrics(
    path: str | Path,
    *,
    exp: str,
    seed: int,
    cfg: Mapping[str, Any],
    metrics: Mapping[str, float],
    checks: Mapping[str, bool],
    runtime_sec: float,
    device: torch.device,
    n_paths: int,
) -> dict[str, Any]:
    """写 spec §5.2 的 metrics.json；passed = checks 全为真。"""
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "exp": exp,
        "seed": int(seed),
        **git_provenance(),
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "runtime_sec": round(float(runtime_sec), 3),
        "device": str(device),
        "torch_version": torch.__version__,
        "n_paths": int(n_paths),
        "config_sha256": digest_config(dict(cfg)),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "checks": {key: bool(value) for key, value in checks.items()},
    }
    payload["passed"] = bool(payload["checks"]) and all(payload["checks"].values())
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
