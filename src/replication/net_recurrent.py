"""半递归网络（论文 §5.1）：δ_k = F_k(I_k, δ_{k−1})。

- 每个时点 k 一套独立权重（论文："these are different for each k"）；
- L = 3：输入 → N1 = N2 = d+15 → 输出 d；ReLU，隐层激活前 batch norm；
- 输入层宽度 f + d（spec §5.1 冻结；论文是 N0 = 2d，差异记 documented deviation）；
- simpler 变体不喂 δ_{k−1}（E1 的架构对照）。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.policy.policy import Policy


def _norm(kind: str, n_hidden: int) -> nn.Module:
    kind = kind.lower()
    if kind in {"batch", "batch_norm", "bn"}:
        return nn.BatchNorm1d(n_hidden)
    if kind in {"layer", "layer_norm", "ln"}:
        return nn.LayerNorm(n_hidden)
    if kind in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"unknown norm kind: {kind}")


class StepMLP(nn.Module):
    """单步前馈：Linear → norm → ReLU → Linear → norm → ReLU → Linear。

    norm='batch' 是论文做法；'layer' 逐样本归一化，训练/评估行为一致（无 running stats），
    用于避免"训练用 batch 统计、评估用 running 统计"导致的递归失稳。
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int, norm: str = "batch") -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, n_out)
        self.bn1 = _norm(norm, n_hidden)
        self.bn2 = _norm(norm, n_hidden)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return self.out(x)


class SemiRecurrentPolicy(nn.Module, Policy):
    """逐时点独立权重的半递归策略；forward 单步、forward_path 批量。

    recurrent=True：输入 [x_k, δ_{k−1}]（论文做法）；False：只喂 x_k（simpler 对照）。
    """

    def __init__(
        self,
        n_features: int,
        d: int = 2,
        n_steps: int = 30,
        hidden: int | None = None,
        recurrent: bool = True,
        batch_norm: bool = True,
        norm: str | None = None,
        out_scale: float | list[float] = 1.0,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.d = d
        self.n_steps = n_steps
        self.recurrent = recurrent
        self.norm_kind = norm if norm is not None else ("batch" if batch_norm else "none")
        n_hidden = int(hidden if hidden is not None else d + 15)
        n_in = n_features + d if recurrent else n_features
        self.steps = nn.ModuleList(
            [StepMLP(n_in, n_hidden, d, norm=self.norm_kind) for _ in range(n_steps)]
        )
        # 输出尺度：δ 的量级差 3 个数量级（标的 ~O(1)，方差互换 ~O(10^2)），
        # Adam 的步长上限是 lr，直接把 500 学出来极慢，因此对输出做固定重参数化。
        scale = torch.tensor(out_scale, dtype=torch.float64).reshape(-1)
        if scale.numel() == 1:
            scale = scale.repeat(d)
        if scale.numel() != d:
            raise ValueError(f"out_scale must be scalar or length {d}, got {out_scale}")
        self.register_buffer("out_scale", scale)
        self.reset()

    def reset(self) -> None:
        self._prev: Tensor | None = None
        self._k = 0

    def forward(self, x_k: Tensor) -> Tensor:
        """单步：(B, f) → (B, d)；内部维护 δ_{k−1}，reset() 清零。"""
        if self._k >= self.n_steps:
            raise RuntimeError(f"forward 调用超过 n_steps={self.n_steps}")
        net = self.steps[self._k]
        if self.recurrent:
            if self._prev is None:
                self._prev = x_k.new_zeros(x_k.shape[0], self.d)
            # 递归反馈用**原始尺度**（δ/out_scale），否则方差腿的 δ2~500 会以巨大数值
            # 进入第一层，训练直接发散。
            inp = torch.cat([x_k, self._prev / self.out_scale], dim=-1)
        else:
            inp = x_k
        out = net(inp) * self.out_scale
        self._prev = out
        self._k += 1
        return out

    def forward_path(self, x: Tensor) -> Tensor:
        """批量：(B, n, f) → (B, n, d)，不改内部状态（训练/评估用）。"""
        batch, n, _ = x.shape
        prev = x.new_zeros(batch, self.d)
        outputs = []
        for k in range(n):
            net = self.steps[k]
            inp = torch.cat([x[:, k], prev / self.out_scale], dim=-1) if self.recurrent else x[:, k]
            prev = net(inp) * self.out_scale
            outputs.append(prev)
        return torch.stack(outputs, dim=1)
