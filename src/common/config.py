"""配置加载、校验与运行时快照（spec §3.1 / §4）。

YAML -> 深合并 DEFAULTS -> 校验 -> config.yaml 快照（合并后的所有默认值都在里面）。
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# 实验 id 白名单，与 docs/spec.md §3.1 一一对应；新增实验先改文档。
EXP_IDS: tuple[str, ...] = (
    "e1",
    "e2",
    "e2cs",
    "e3",
    "e4bs",
    "e4h",
    "e4low",
    "e1d1",
    "debug",
    "hk",
)

DEFAULTS: dict[str, Any] = {
    "exp": None,
    "kind": "smoke",
    "seeds": [0],
    "device": "auto",
    "out_root": "results",
    "n_steps": 30,
    "batch": 256,
    "eps": 0.0,
    "alpha": 0.5,
    "policy": {"name": "constant", "value": 0.0},
    "market": {
        "model": "gbm",
        "s0": 100.0,
        "k": 100.0,
        "t": 30.0 / 365.0,
        "sigma": 0.2,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并 override 进 base 的副本；返回新字典，不改入参。"""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """读 YAML 并与 DEFAULTS 合并，返回带 _source 的运行时配置。"""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"config not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"config must be a mapping, got {type(raw).__name__}: {source}")
    cfg = deep_merge(DEFAULTS, raw)
    validate_config(cfg)
    cfg["_source"] = str(source.resolve())
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """结构校验；不合规直接抛 ValueError（比跑半小时再报错强）。"""
    exp = cfg.get("exp")
    if exp not in EXP_IDS:
        raise ValueError(f"exp={exp!r} 不在白名单，先改 docs/spec.md §3.1；合法值：{EXP_IDS}")
    n_steps = cfg.get("n_steps")
    if not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError(f"n_steps must be a positive int, got {n_steps!r}")
    batch = cfg.get("batch")
    if not isinstance(batch, int) or batch < 1:
        raise ValueError(f"batch must be a positive int, got {batch!r}")
    alpha = float(cfg.get("alpha", 0.5))
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if float(cfg.get("eps", 0.0)) < 0.0:
        raise ValueError(f"eps must be >= 0, got {cfg['eps']!r}")


def resolve_seeds(cfg: dict[str, Any], seed: int | None = None) -> list[int]:
    """命令行 --seed 覆盖 config；否则用 seeds；都没有则 [0]。"""
    if seed is not None:
        return [int(seed)]
    seeds = cfg.get("seeds") or []
    if isinstance(seeds, int):
        seeds = [seeds]
    return [int(s) for s in seeds] or [0]


def runtime_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """去掉内部键（以下划线开头）后的快照内容。"""
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def dump_runtime_config(cfg: dict[str, Any], path: str | Path) -> Path:
    """把合并后的实际配置写成 config.yaml（spec §4）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(runtime_config(cfg), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


def digest_config(cfg: dict[str, Any]) -> str:
    """运行配置的 sha256，写进 metrics.json 的 config_sha256。"""
    payload = json.dumps(runtime_config(cfg), sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
