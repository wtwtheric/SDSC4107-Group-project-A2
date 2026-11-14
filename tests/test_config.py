from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from src.common.config import (
    DEFAULTS,
    deep_merge,
    digest_config,
    dump_runtime_config,
    load_config,
    resolve_seeds,
    runtime_config,
)

DEBUG_YAML = """\
exp: debug
kind: smoke
batch: 8
"""


def write_cfg(tmp_path: Path, text: str = DEBUG_YAML) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_deep_merge_nested() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 0}
    out = deep_merge(base, {"a": {"y": 9, "z": 3}})
    assert out == {"a": {"x": 1, "y": 9, "z": 3}, "b": 0}
    assert base["a"]["y"] == 2


def test_load_config_merges_defaults(tmp_path: Path) -> None:
    cfg = load_config(write_cfg(tmp_path))
    assert cfg["exp"] == "debug"
    assert cfg["batch"] == 8
    assert cfg["market"]["s0"] == DEFAULTS["market"]["s0"]
    assert cfg["n_steps"] == DEFAULTS["n_steps"]
    assert cfg["_source"].endswith("cfg.yaml")


def test_unknown_exp_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="白名单"):
        load_config(write_cfg(tmp_path, "exp: e99\n"))


def test_bad_alpha_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="alpha"):
        load_config(write_cfg(tmp_path, "exp: debug\nalpha: 1.5\n"))


def test_dump_runtime_config_roundtrip(tmp_path: Path) -> None:
    cfg = load_config(write_cfg(tmp_path))
    target = dump_runtime_config(cfg, tmp_path / "out" / "config.yaml")
    loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert loaded == runtime_config(cfg)
    assert "_source" not in loaded


def test_digest_stable_and_sensitive(tmp_path: Path) -> None:
    cfg = load_config(write_cfg(tmp_path))
    same = load_config(write_cfg(tmp_path))
    assert digest_config(cfg) == digest_config(same)
    cfg["batch"] = cfg["batch"] + 1
    assert digest_config(cfg) != digest_config(same)


def test_resolve_seeds_override() -> None:
    assert resolve_seeds({"seeds": [0, 1, 2]}, 7) == [7]
    assert resolve_seeds({"seeds": [0, 1, 2]}) == [0, 1, 2]
    assert resolve_seeds({"seeds": 3}) == [3]
    assert resolve_seeds({}) == [0]
