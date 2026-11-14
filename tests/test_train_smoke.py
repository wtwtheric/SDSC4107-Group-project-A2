from __future__ import annotations

import json

import numpy as np
from src.common.config import REPO_ROOT
from src.replication.train import main


def test_debug_run_writes_all_artifacts(tmp_path) -> None:
    cfg_path = REPO_ROOT / "configs" / "exp" / "debug.yaml"
    rc = main(["--config", str(cfg_path), "--out", str(tmp_path), "--seed", "0"])
    assert rc == 0

    run = tmp_path / "debug" / "0"
    for name in ("metrics.json", "config.yaml", "run.log", "pnl.npz"):
        assert (run / name).exists(), name

    payload = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["exp"] == "debug"
    assert payload["seed"] == 0
    assert payload["passed"] is True
    assert payload["n_paths"] == 512
    assert payload["runtime_sec"] > 0.0
    assert payload["device"] == "cpu"
    assert {"mean_loss", "cvar50", "cvar99"} <= set(payload["metrics"])

    with np.load(run / "pnl.npz") as data:
        assert data["terminal_loss"].shape == (512,)

    log_text = (run / "run.log").read_text(encoding="utf-8")
    assert "exp=debug seed=0" in log_text
