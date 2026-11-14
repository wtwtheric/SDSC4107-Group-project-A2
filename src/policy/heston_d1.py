"""A1 → B2 的冻结策略：policy_heston_d1（Heston 世界、只对冲标的）。

用法（B2 回测）：

    from src.policy.heston_d1 import load
    policy = load()                      # -> SemiRecurrentPolicy，已 eval()
    policy.reset()
    for k in range(n_steps):
        delta = policy(features[:, k])   # features: (B, 2) = [log(S_k/S_0), t_k/T]
        # delta: (B, 1)，单位是"份标的"

权重由 scripts/export_policy_d1.py 从 results/e1d1/<seed>/policy.pt 导出，
checkpoint 里带 meta（特征定义、市场参数、seed、git commit、评估指标）。

注意：这个策略训练时**只看标的**（d=1、无波动率状态），所以在港股上做迁移时，
它的输入只需要标的与时间——这正是 B2 回测的做法；不要传入额外的波动率特征。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.replication.net_recurrent import SemiRecurrentPolicy

WEIGHTS = Path(__file__).with_name("weights") / "heston_d1.pt"


def read_checkpoint(path: str | Path = WEIGHTS) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def build_from_checkpoint(
    payload: dict[str, Any], device: torch.device | str = "cpu"
) -> SemiRecurrentPolicy:
    meta = payload["meta"]
    policy = SemiRecurrentPolicy(
        n_features=int(meta["n_features"]),
        d=int(meta["d"]),
        n_steps=int(meta["n_steps"]),
        hidden=int(meta["hidden"]),
        recurrent=bool(meta["recurrent"]),
        batch_norm=bool(meta.get("batch_norm", True)),
        norm=meta.get("norm"),
        out_scale=meta.get("out_scale", 1.0),
    )
    policy.load_state_dict(payload["state_dict"])
    policy.to(device=device, dtype=torch.float64).eval()
    for p in policy.parameters():  # 冻结：回测里不该有梯度
        p.requires_grad_(False)
    return policy


def load(path: str | Path = WEIGHTS, device: torch.device | str = "cpu") -> SemiRecurrentPolicy:
    """加载冻结权重；不可用时抛 FileNotFoundError（提示先跑导出脚本）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在；先跑 scripts/export_policy_d1.py")
    return build_from_checkpoint(read_checkpoint(path), device=device)
