"""共享类型与结果路径（spec §4）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class RunPaths:
    """一次 (exp, seed) 运行的落盘位置；目录名与 spec §3.1 一致。"""

    out_root: Path
    exp: str
    seed: int

    @property
    def run_dir(self) -> Path:
        return self.out_root / self.exp / str(self.seed)

    @property
    def metrics(self) -> Path:
        return self.run_dir / "metrics.json"

    @property
    def config(self) -> Path:
        return self.run_dir / "config.yaml"

    @property
    def log(self) -> Path:
        return self.run_dir / "run.log"

    @property
    def pnl(self) -> Path:
        return self.run_dir / "pnl.npz"

    def mkdir(self) -> RunPaths:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self
