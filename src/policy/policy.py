"""策略接口（spec §5.1 冻结）：reset() 每个 episode 调一次，forward(x_k) -> position。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Policy(Protocol):
    """A 组所有策略（网络与基准）都满足这个接口；B 组只认 forward。"""

    def reset(self) -> None:
        """清空 recurrent 隐状态；非 recurrent 策略可以为空实现。"""
        ...

    def forward(self, x_k: torch.Tensor) -> torch.Tensor:
        """x_k: (B, f) -> position: (B, d)。"""
        ...


class ConstantPolicy(torch.nn.Module):
    """位置恒为常数的占位策略，只用于 G0 冒烟，不产生任何实验结论。"""

    def __init__(self, d: int = 1, value: float = 0.0) -> None:
        super().__init__()
        self.d = int(d)
        self.value = float(value)

    def reset(self) -> None:
        return None

    def forward(self, x_k: torch.Tensor) -> torch.Tensor:
        batch = x_k.shape[0]
        return torch.full(
            (batch, self.d),
            self.value,
            dtype=x_k.dtype,
            device=x_k.device,
        )
