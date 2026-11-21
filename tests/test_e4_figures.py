"""Regression tests for E4 power-law figures and sensitivity calculations."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from scripts.a2.make_e4_figures import THEORETICAL_SLOPE, load_grid, make_figures, regression


def test_regression_recovers_exact_power_law() -> None:
    eps = np.array([2.0**-k for k in (10, 9, 8, 7, 6)])
    increments = 17 * eps**THEORETICAL_SLOPE
    fit = regression(eps, increments)
    assert fit["slope"] == pytest.approx(THEORETICAL_SLOPE, abs=1e-12)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)
    assert fit["fixed_r2"] == pytest.approx(1.0, abs=1e-12)


def _save_grid(path, *, exp: str, seed: int, slope: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for k in (6, 7, 8, 9, 10):
        eps = 2.0**-k
        inc = 10.0 * eps**slope
        records.append(
            {
                "exp": exp,
                "seed": seed,
                "passed": True,
                "metrics": {
                    "eps": eps,
                    "price_eps": 2.0 + inc,
                    "price0": 2.0,
                    "price_increment": inc,
                },
            }
        )
    path.write_text(json.dumps(records), encoding="utf-8")


def test_load_grid_rejects_unreliable_price_increment(tmp_path) -> None:
    path = tmp_path / "bad.json"
    _save_grid(path, exp="e4bs", seed=0, slope=0.7)
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows[0]["metrics"]["price_increment"] = -0.5
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid or duplicate epsilon"):
        load_grid(path, exp="e4bs", seed=0)


def test_e4_figures_and_tables_use_separate_per_seed_slopes(tmp_path) -> None:
    for exp in ("e4bs", "e4h"):
        for seed in (0, 1):
            _save_grid(
                tmp_path / "backups" / exp / f"seed_{seed}_epsilon_summary.json",
                exp=exp,
                seed=seed,
                slope=0.66 + 0.02 * seed,
            )
    output = make_figures(
        data_root=tmp_path / "backups",
        figures_dir=tmp_path / "figures",
        report_dir=tmp_path / "reports",
        seeds=[0, 1],
    )
    assert len(output) == 6
    assert all(path.stat().st_size > 0 for path in output)
    with (tmp_path / "reports" / "e4_epsilon_sensitivity.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20  # 2 models x 2 seeds x 5 epsilon subsets
    with (tmp_path / "reports" / "e4_seed_slopes.csv").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == 2
    assert all(float(row["slope_mean"]) == pytest.approx(0.67) for row in summary)
    assert all(float(row["slope_sd"]) > 0 for row in summary)
