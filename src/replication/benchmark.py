"""基准对冲（benchmark）：Heston 模型内的解析 delta 面 + 模型对冲损益。

为什么是"模型内基准"：论文 §5.2 的基线是 Heston 模型自身的 delta（用 Quantlib 算的
BS 式 delta），不是 BS 模型。这里复算成 (S, v) 上的二维面：每个时点 k 用剩余期限 τ_k
的 Heston 看涨价 C(S, v, τ_k)，δ1 = ∂C/∂S，δ2 = ∂C/∂v（方差互换价格 S² = m_k·v，
m_k = (1−e^{−κτ})/(κτ)，故 ∂C/∂S² = (∂C/∂v)/m_k）。

数值：价格面用 pricing.heston_call_prices（u = tanθ 换元，整张网格 30 个期限约 1 分钟），
δ 由 np.gradient 得到，再用 RegularGridInterpolator(cubic) 在路径上取值（向量化，1e5 条路径
× 30 步约数秒）。整张面缓存到 results/b0/heston_surface.npz，重复实验不再重算。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm
from torch import Tensor

from src.replication.env_bs import BSPaths, BSWorld
from src.replication.env_heston import HestonPaths, HestonWorld
from src.replication.pricing import HestonParams, heston_call_prices

# 网格要覆盖模拟路径的尾部（实测 v 最高 1.67、S 最低 40），否则插值外推会把 δ2 放大到 1e6。
# S：30–160 步长 1，ATM 90–110 加密到 0.25（最后一步 τ=1/365 的 delta 过渡带 ≈ 1）。
GRID_SPOT = np.unique(np.concatenate([np.linspace(30.0, 160.0, 131), np.linspace(90.0, 110.0, 81)]))
# v：0–0.12 步长 0.002（主体），0.12–0.4 步长 0.01，0.4–1.8 步长 0.05（尾部）。
GRID_VAR = np.unique(
    np.concatenate(
        [np.linspace(0.0, 0.12, 61), np.linspace(0.12, 0.4, 29), np.linspace(0.4, 1.8, 29)]
    )
)
SURFACE_CACHE = Path("results/b0/heston_surface_v2.npz")


@dataclass
class HestonBenchmark:
    """Heston 模型 delta 基准；positions() 给出 (B, n, d) 的对冲头寸。

    cache=None 且传入粗网格时完全不落盘，供单测使用（全网格建面约 2.5 分钟）。
    """

    market: HestonWorld
    cache: Path | None = SURFACE_CACHE
    grid_spot: np.ndarray | None = None
    grid_var: np.ndarray | None = None

    def __post_init__(self) -> None:
        m = self.market.market
        self.grid_spot = GRID_SPOT if self.grid_spot is None else np.asarray(self.grid_spot)
        self.grid_var = GRID_VAR if self.grid_var is None else np.asarray(self.grid_var)
        self.params = HestonParams(
            kappa=m.kappa,
            theta=m.theta,
            sigma=m.sigma,
            rho=m.rho,
            v0=m.v0,
            s0=m.s0,
            strike=m.strike,
            maturity=m.maturity,
        )
        self.n_steps = m.n_steps
        self.taus = np.array([m.maturity - k * m.dt for k in range(m.n_steps)])
        self._call: list[RegularGridInterpolator] = []
        self._d1: list[RegularGridInterpolator] = []
        self._dv: list[RegularGridInterpolator] = []
        self._build()

    # ---------------------------------------------------------------- 建面
    def _build(self) -> None:
        cache = self.cache
        key = (
            f"{self.params.kappa}_{self.params.theta}_{self.params.sigma}_"
            f"{self.params.rho}_{self.params.maturity}"
        )
        data: dict[str, np.ndarray] = {}
        loaded_ok = False
        if cache is not None and cache.exists():
            loaded = np.load(cache)
            if str(loaded.get("key", "")) == key:
                data = {k: loaded[k] for k in loaded.files if k != "key"}
                loaded_ok = True
                print(f"[benchmark] loaded cached surface: {cache}")
        sg, vg = np.meshgrid(self.grid_spot, self.grid_var, indexing="ij")
        for i, tau in enumerate(self.taus):
            name_c, name_s, name_v = f"c{i}", f"ds{i}", f"dv{i}"
            if name_c not in data:
                price = heston_call_prices(sg, vg, tau, self.params)
                price = np.clip(price, 0.0, None)
                data[name_c] = price
                data[name_s] = np.gradient(price, self.grid_spot, axis=0)
                data[name_v] = np.gradient(price, self.grid_var, axis=1)
            self._call.append(self._interp(data[name_c]))
            self._d1.append(self._interp(data[name_s]))
            self._dv.append(self._interp(data[name_v]))

        if cache is not None and not loaded_ok:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, key=np.array(key), **data)
            print(f"[benchmark] built and cached surface -> {cache}")
        self._data = data

    def _interp(self, values: np.ndarray) -> RegularGridInterpolator:
        return RegularGridInterpolator(
            (self.grid_spot, self.grid_var),
            values,
            method="cubic",
            bounds_error=False,
            fill_value=None,
        )

    # ---------------------------------------------------------------- 取值
    def call(self, spot: np.ndarray, variance: np.ndarray, k: int) -> np.ndarray:
        shape = np.shape(spot)
        pts = np.column_stack([np.ravel(spot), np.ravel(variance)])
        return self._call[k](pts).reshape(shape)

    def delta(
        self, spot: np.ndarray, variance: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回 (δ1, δ2)，δ2 已换成对"方差互换价格"的导数。"""
        shape = np.shape(spot)
        pts = np.column_stack([np.ravel(spot), np.ravel(variance)])
        d1 = self._d1[k](pts).reshape(shape)
        dv = self._dv[k](pts).reshape(shape)
        # 论文 (5.6)：δ2 = ∂_v u / ∂_v L(t, v)，∂_v L = (1 − e^{−κ(T−t)})/κ。
        # 必须是除法：S² = ∫V ds + L(t,V)，dS² 的 dV 项系数是 ∂_v L。
        factor = (1.0 - np.exp(-self.params.kappa * float(self.taus[k]))) / self.params.kappa
        return d1, dv / factor

    def positions(self, paths: HestonPaths, d: int | None = None) -> Tensor:
        d = self.market.market.d if d is None else d
        spot = paths.spot.detach().cpu().numpy()
        var = paths.variance.detach().cpu().numpy()
        out = np.empty((spot.shape[0], self.n_steps, d))
        for k in range(self.n_steps):
            d1, d2 = self.delta(spot[:, k], var[:, k], k)
            out[:, k, 0] = d1
            if d == 2:
                out[:, k, 1] = d2
        return torch.as_tensor(out, dtype=paths.spot.dtype)


@dataclass
class BSHedge:
    """BS 世界（E4-BS）的基准：解析 delta。"""

    world: BSWorld

    def positions(self, paths: BSPaths) -> Tensor:
        m = self.world.market
        spot = paths.spot.detach().cpu().numpy()
        out = np.empty((spot.shape[0], m.n_steps, 1))
        for k in range(m.n_steps):
            tau = m.maturity - k * m.dt
            dd = (np.log(spot[:, k] / m.strike) + 0.5 * m.sigma**2 * tau) / (m.sigma * np.sqrt(tau))
            out[:, k, 0] = norm.cdf(dd)
        return torch.as_tensor(out, dtype=paths.spot.dtype)
