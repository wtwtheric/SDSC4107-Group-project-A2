"""Tests for reproducible E2cs diagnostic metrics and publication-style figures."""

from __future__ import annotations

import numpy as np
import pytest
from scripts.a2.diagnose_e2cs import _metrics
from scripts.a2.make_e2cs_figures import (
    _comparable_norm,
    _symmetric_norm,
    plot_surface_pairs,
)


def test_stock_grid_metrics_detect_flattening_and_peak_mismatch() -> None:
    spots = np.array([99.0, 100.0, 101.0])
    variances = np.array([0.02, 0.04])
    model = np.array([[0.0, 0.1], [0.7, 0.5], [0.0, 0.1]])
    nn = np.full_like(model, 0.10)
    metrics = _metrics(model, nn, spots, variances)
    assert metrics["model_peak_spot"] == 100.0
    assert metrics["model_peak_variance"] == 0.02
    assert metrics["model_peak_height"] == pytest.approx(0.7)
    assert metrics["policy_at_model_peak"] == pytest.approx(0.1)
    assert metrics["difference_at_model_peak"] == pytest.approx(-0.6)
    assert metrics["low_variance_model_spot_range"] == pytest.approx(0.7)
    assert metrics["low_variance_policy_spot_range"] == pytest.approx(0.0)
    assert metrics["correlation"] is None


def test_comparable_colors_are_shared_and_difference_centered_on_white() -> None:
    model = np.array([[0.0, 0.1], [0.8, 0.5]])
    nn95 = np.array([[0.05, 0.1], [0.7, 0.45]])
    nn99 = np.array([[0.02, 0.05], [0.2, 0.17]])
    norm = _comparable_norm([model, nn95, nn99])
    diff_norm = _symmetric_norm([nn95 - model, nn99 - model])
    assert norm.vmin == pytest.approx(0.0)
    assert norm.vmax == pytest.approx(0.8)
    assert diff_norm.vcenter == 0.0
    assert diff_norm.vmin == pytest.approx(-diff_norm.vmax)


def test_render_both_alphas_from_existing_grids(tmp_path) -> None:
    spots = np.array([99.0, 100.0, 101.0])
    variances = np.array([0.02, 0.04])
    model = np.array([[0.0, 0.1], [0.7, 0.5], [0.0, 0.1]])
    entries = {}
    for alpha, factor in (("0.95", 0.9), ("0.99", 0.4)):
        entries[alpha] = {
            "model_spot_grid": spots,
            "model_variance_grid": variances,
            "model_stock_delta_grid": model,
            "policy_stock_delta_grid": model * factor,
        }
    output = plot_surface_pairs(entries, tmp_path)
    assert len(output) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in output)
