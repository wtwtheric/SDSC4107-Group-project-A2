"""Tests for E2/E3 paper-fidelity simulator and input controls."""

from __future__ import annotations

import torch

from src.common.seed import seed_everything
from src.replication.env_heston import (
    HestonMarket,
    HestonWorld,
    _sample_exact_cir_transition,
)


def test_exact_cir_transition_is_nonnegative_and_has_correct_mean() -> None:
    seed_everything(12345)
    n = 50_000
    v0 = torch.full((n,), 0.04, dtype=torch.float64)
    out = _sample_exact_cir_transition(
        v0,
        kappa=1.0,
        theta=0.04,
        sigma=2.0,
        dt=1.0 / 365.0,
    )
    assert torch.isfinite(out).all()
    assert bool((out >= 0.0).all())
    # v0 == theta, so E[V_{t+dt}|V_t=theta] == theta exactly.
    assert abs(float(out.mean()) - 0.04) < 0.002


def test_paper_simulator_produces_finite_paths_and_expected_initial_varswap() -> None:
    seed_everything(222)
    market = HestonMarket(simulator="paper_sbk", feature_set="current")
    world = HestonWorld(market, device="cpu")
    paths = world.simulate(1024)
    assert paths.spot.shape == (1024, 31)
    assert paths.variance.shape == (1024, 31)
    assert paths.varswap.shape == (1024, 31)
    assert torch.isfinite(paths.spot).all()
    assert torch.isfinite(paths.variance).all()
    assert torch.isfinite(paths.varswap).all()
    assert bool((paths.spot > 0.0).all())
    assert bool((paths.variance >= 0.0).all())
    expected_s20 = market.theta * market.maturity
    torch.testing.assert_close(
        paths.varswap[:, 0],
        torch.full((1024,), expected_s20, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_paper_feature_set_is_exact_log_spot_and_variance() -> None:
    seed_everything(333)
    market = HestonMarket(simulator="qem", feature_set="paper")
    world = HestonWorld(market, device="cpu")
    paths = world.simulate(128)
    features = world.features(paths)
    assert features.shape == (128, 30, 2)
    torch.testing.assert_close(features[..., 0], torch.log(paths.spot[:, :30]))
    torch.testing.assert_close(features[..., 1], paths.variance[:, :30])


def test_current_feature_set_shape_is_unchanged() -> None:
    seed_everything(444)
    market = HestonMarket(simulator="qem", feature_set="current")
    world = HestonWorld(market, device="cpu")
    features = world.features(world.simulate(64))
    assert market.n_features == 3
    assert features.shape == (64, 30, 3)


def test_paper_feature_set_reports_two_features() -> None:
    assert HestonMarket(feature_set="paper").n_features == 2
