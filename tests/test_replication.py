"""replication 层单测：定价器、两套环境、策略、损失、基准对冲。"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from src.common.metrics import cvar
from src.common.seed import seed_everything
from src.replication.benchmark import BSHedge, HestonBenchmark
from src.replication.env_bs import BSMarket, BSWorld
from src.replication.env_heston import HestonMarket, HestonWorld
from src.replication.losses import avg_cvar, entropic_loss
from src.replication.net_recurrent import SemiRecurrentPolicy
from src.replication.pricing import (
    PAPER,
    bs_call,
    heston_call_prices,
    heston_call_scalar,
    varswap_price_total,
    varswap_price_total_dv,
)


def _heston_world(d: int = 2, eps: float = 0.0, n_paths_market=None) -> HestonWorld:
    market = HestonMarket(eps=eps, d=d)
    return HestonWorld(market, device=torch.device("cpu"))


# ---------------------------------------------------------------- 定价器
def test_heston_prices_match_quad_reference() -> None:
    spot = np.array([90.0, 100.0, 110.0])
    var = np.array([0.01, 0.04, 0.09])
    fast = heston_call_prices(spot, var, PAPER.maturity)
    ref = np.array(
        [heston_call_scalar(s, v, PAPER.maturity) for s, v in zip(spot, var, strict=True)]
    )
    # 深虚值 (S=90, v=0.01) 的尾部积分残差约 2e-5，其余点 ~1e-13
    assert np.allclose(fast, ref, atol=2e-5)


def test_heston_price_shape_and_monotonicity() -> None:
    spot = np.linspace(85.0, 115.0, 13)
    var = np.full_like(spot, 0.04)
    price = heston_call_prices(spot, var, PAPER.maturity)
    assert price.shape == spot.shape
    assert np.all(np.diff(price) > 0.0)  # call 价随 S 单调增
    assert np.all(price >= 0.0)


def test_varswap_price_is_total_expected_variance() -> None:
    t = 0.0
    assert varswap_price_total(0.04, t, PAPER.maturity) == pytest.approx(0.04 * PAPER.maturity)
    # 到期时 L(T, v) = 0，dL/dv 也归零
    assert varswap_price_total(0.3, PAPER.maturity, PAPER.maturity) == pytest.approx(0.0)
    assert varswap_price_total_dv(PAPER.maturity, PAPER.maturity) == pytest.approx(0.0)
    # ∂_v L 在 t=0 即 (1−e^{−κT})/κ
    expect = (1.0 - np.exp(-PAPER.kappa * PAPER.maturity)) / PAPER.kappa
    assert varswap_price_total_dv(t, PAPER.maturity) == pytest.approx(expect)


def test_bs_call_matches_analytic_with_intrinsic() -> None:
    assert bs_call(100.0, 100.0, 0.0, 0.2) == pytest.approx(0.0, abs=1e-9)
    assert bs_call(120.0, 100.0, 0.2, 0.2) >= 20.0


# ---------------------------------------------------------------- Heston 环境
def test_heston_shapes_and_static_features() -> None:
    world = _heston_world()
    paths = world.simulate(64)
    m = world.market
    assert paths.spot.shape == (64, m.n_steps + 1)
    assert paths.variance.shape == (64, m.n_steps + 1)
    assert paths.varswap.shape == (64, m.n_steps + 1)
    feats = world.features(paths)
    assert feats.shape == (64, m.n_steps, 3)
    assert torch.allclose(feats[:, 0, 0], torch.zeros(64, dtype=feats.dtype))
    assert torch.allclose(feats[:, 0, 2], torch.ones(64, dtype=feats.dtype))
    assert torch.allclose(paths.spot[:, 0], torch.full((64,), 100.0, dtype=paths.spot.dtype))
    prices = world.prices(paths)
    assert prices.shape == (64, m.n_steps + 1, 2)
    assert torch.allclose(paths.spot[:, 0], prices[:, 0, 0])
    # t=0 的方差互换价格 = bT
    assert float(paths.varswap[0, 0]) == pytest.approx(0.04 * m.maturity, rel=1e-6)


def test_zero_positions_loss_equals_payoff() -> None:
    world = _heston_world()
    paths = world.simulate(32)
    zeros = torch.zeros(32, world.market.n_steps, 2, dtype=torch.float64)
    loss = world.terminal_loss(zeros, paths)
    assert torch.allclose(loss, world.payoff(paths))


def test_cost_matches_manual_sum() -> None:
    eps = 0.02
    world = _heston_world(eps=eps)
    paths = world.simulate(16)
    positions = torch.rand(16, world.market.n_steps, 2, dtype=torch.float64)
    cost = world.cost(positions, paths)

    spot = paths.spot.detach().numpy()
    varswap = paths.varswap.detach().numpy()
    pos = positions.detach().numpy()
    manual = np.zeros(16)
    for k in range(world.market.n_steps):
        prev = np.zeros_like(pos[:, 0])
        for i in range(k):
            prev = pos[:, i]
        trade = np.abs(pos[:, k] - prev)
        manual += eps * (trade[:, 0] * spot[:, k] + trade[:, 1] * varswap[:, k])
    assert np.allclose(cost.detach().numpy(), manual, rtol=1e-10)


def test_losses_wrappers() -> None:
    seed_everything(0)
    loss = torch.randn(4096, dtype=torch.float64) + 1.0
    assert float(avg_cvar(loss, 0.99)) == pytest.approx(float(cvar(loss, 0.99)))
    assert float(cvar(loss.detach().numpy(), 0.99)) == pytest.approx(float(cvar(loss, 0.99)))
    assert float(entropic_loss(loss, 1.0)) > 0.0
    assert float(entropic_loss(loss * 0.0, 1.0)) == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------- 策略
def test_policy_forward_path_matches_stepwise() -> None:
    seed_everything(0)
    policy = SemiRecurrentPolicy(n_features=3, d=2, n_steps=5, hidden=17, recurrent=True).double()
    x = torch.randn(8, 5, 3, dtype=torch.float64)
    path = policy.forward_path(x)
    policy.reset()
    steps = [policy(x[:, k]) for k in range(5)]
    stacked = torch.stack(steps, dim=1)
    assert path.shape == (8, 5, 2)
    assert torch.allclose(path, stacked)


def test_policy_reset_clears_previous_position() -> None:
    seed_everything(0)
    policy = SemiRecurrentPolicy(n_features=2, d=1, n_steps=3, hidden=16).double()
    x = torch.randn(4, 2, dtype=torch.float64)
    policy.reset()
    first = policy(x).clone()
    policy.reset()
    second = policy(x)
    assert torch.allclose(first, second)


# ---------------------------------------------------------------- BS 环境
def test_bs_world_mc_matches_analytic() -> None:
    seed_everything(12345)
    world = BSWorld(BSMarket(), device=torch.device("cpu"))
    n_chunks, chunk = 5, 200_000
    total = 0.0
    for _ in range(n_chunks):
        paths = world.simulate(chunk)
        total += float(world.payoff(paths).sum())
    mc = total / (n_chunks * chunk)
    analytic = float(world.analytic_price())
    assert abs(mc - analytic) / analytic < 0.01


def test_bs_shapes_and_features() -> None:
    world = BSWorld(BSMarket(), device=torch.device("cpu"))
    paths = world.simulate(16)
    assert paths.spot.shape == (16, world.market.n_steps + 1)
    assert world.features(paths).shape == (16, world.market.n_steps, 2)
    assert world.prices(paths).shape == (16, world.market.n_steps + 1, 1)


# ---------------------------------------------------------------- 基准对冲
@pytest.fixture(scope="module")
def coarse_benchmark() -> HestonBenchmark:
    world = _heston_world()
    return HestonBenchmark(
        world,
        cache=None,
        grid_spot=np.linspace(60.0, 150.0, 91),
        grid_var=np.linspace(0.002, 0.3, 61),
    )


def test_benchmark_deltas_are_sane(coarse_benchmark: HestonBenchmark) -> None:
    d1, d2 = coarse_benchmark.delta(np.array([100.0]), np.array([0.04]), k=0)
    assert 0.4 < float(d1[0]) < 0.7  # ATM delta
    assert float(d2[0]) > 0.0  # call 对 var 的导数 > 0
    # δ1 随 S 单调增
    spot = np.linspace(85.0, 115.0, 7)
    d1s, _ = coarse_benchmark.delta(spot, np.full_like(spot, 0.04), k=10)
    assert np.all(np.diff(d1s) > 0.0)


def test_benchmark_hedge_prices_the_call(coarse_benchmark: HestonBenchmark) -> None:
    world = coarse_benchmark.market
    paths = world.simulate(20_000)
    positions = coarse_benchmark.positions(paths)
    assert positions.shape == (20_000, world.market.n_steps, 2)
    loss = world.terminal_loss(positions, paths).detach().numpy()
    # 零成本下 E[L] = q（对冲误差均值为 0）。这里的网格是粗的（S 91 点 / v 61 点），
    # 插值误差会把 E[L] 抬高约 8%，所以只做量级校验；精确对拍由 B0 + 全网格的 fig02 负责。
    assert abs(loss.mean() - 1.6918) < 0.2
    assert cvar(loss, 0.5) > loss.mean()  # CVaR 高于均值


def test_bs_hedge_positions_are_analytic_delta() -> None:
    world = BSWorld(BSMarket(), device=torch.device("cpu"))
    paths = world.simulate(8)
    positions = BSHedge(world).positions(paths)
    assert positions.shape == (8, world.market.n_steps, 1)
    assert torch.all((positions >= 0.0) & (positions <= 1.0))
