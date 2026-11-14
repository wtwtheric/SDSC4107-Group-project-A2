"""训练目标：avg-CVaR 与 entropic（spec §6 A1 任务 2）。

只是 src.common.metrics 的薄封装——全仓库只保留一份 CVaR 实现（spec §5.4 的硬约束）。
"""

from __future__ import annotations

from torch import Tensor

from src.common import metrics


def avg_cvar(loss: Tensor, alpha: float = 0.5) -> Tensor:
    """平均 VaR / CVaR（论文 (5.1)）：最差 (1−α) 比例损失的均值。"""
    return metrics.cvar(loss, alpha)


def entropic_loss(loss: Tensor, lam: float = 1.0) -> Tensor:
    """熵风险测度 ρ(L) = (1/λ) log E[e^{λL}]（论文式 (3.4)，E4 用 λ=1）。"""
    return metrics.entropic(loss, lam)


def indifference_price(loss: Tensor, alpha: float = 0.5) -> float:
    """p0 = CVaR_α(L)；零成本时也等于 q + CVaR_α(L−q)，与 metrics.indifference_price 一致。"""
    return float(metrics.cvar(loss, alpha))


def objective(loss: Tensor, measure: str = "cvar", alpha: float = 0.5) -> Tensor:
    """按名字取目标，训练脚本统一入口。"""
    if measure == "cvar":
        return avg_cvar(loss, alpha)
    if measure == "entropic":
        return entropic_loss(loss, alpha)
    raise ValueError(f"unknown measure={measure!r}; use 'cvar' or 'entropic'")
