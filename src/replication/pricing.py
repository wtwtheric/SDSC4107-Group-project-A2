"""Heston / Black–Scholes 定价（Carr–Madan = Lewis 特征函数积分）。

约定：利率与分红率都为 0（论文 §5.2 的 r = 0，q(α=0.5) = 1.692 可据此复算）。

数值方案（实测过三种，见 docs/method_replication.md §定价器）：
- 固定高斯–勒让德节点在 u ∈ [0, u_max] 上：`1/(u²+1/4)` 的尾部截断在深度虚值角落
  （S=90, v=0.01）会带来 O(0.4) 的绝对误差 → 不可用；
- 本模块用 u = tan θ、θ ∈ [0, π/2) 换元（`du = (1+u²)dθ`，正好抵消 1/u² 尾部），
  尾部只剩 |φ(u)| 的衰减，向量化后整张 (S, v) 网格 30 个期限约 1 分钟；
- `heston_call_scalar` 用 `scipy.integrate.quad_vec` 自适应积分到 ∞，作为参考实现
  （单点调用，仅校验/单测用）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad_vec
from scipy.stats import norm


@dataclass(frozen=True)
class HestonParams:
    """Heston 参数；默认值 = 论文 §5.2（α=1, b=0.04, ρ=−0.7, σ=2, v0=0.04, s0=100）。"""

    kappa: float = 1.0
    theta: float = 0.04
    sigma: float = 2.0
    rho: float = -0.7
    v0: float = 0.04
    s0: float = 100.0
    strike: float = 100.0
    maturity: float = 30.0 / 365.0


PAPER = HestonParams()


def _cf_terms(
    u: np.ndarray, maturity: float, params: HestonParams
) -> tuple[np.ndarray, np.ndarray]:
    """Heston 特征函数的 (c, d)：φ(u) = exp(c + d·v0)，r = q = 0。"""
    u = np.asarray(u, dtype=np.complex128)
    i = 1j
    kappa, theta, sigma, rho = params.kappa, params.theta, params.sigma, params.rho
    root = np.sqrt((rho * sigma * i * u - kappa) ** 2 + sigma**2 * (u**2 + i * u))
    num = kappa - rho * sigma * i * u - root
    den = kappa - rho * sigma * i * u + root
    g = num / den
    decay = np.exp(-root * maturity)
    d_term = (num / sigma**2) * (1.0 - decay) / (1.0 - g * decay)
    c_term = (kappa * theta / sigma**2) * (
        num * maturity - 2.0 * np.log((1.0 - g * decay) / (1.0 - g))
    )
    return c_term, d_term


def heston_cf(u: np.ndarray, maturity: float, params: HestonParams = PAPER) -> np.ndarray:
    """φ(u) = E[exp(i u log(S_T/S_0))]（v = v0）。"""
    c_term, d_term = _cf_terms(u, maturity, params)
    return np.exp(c_term + d_term * params.v0)


def _quadrature(n_u: int, theta_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """u = tan θ 的节点、权重与雅可比：返回 (u, jac)，∫_0^∞ g(u) du ≈ Σ_j g(u_j)·jac_j。"""
    nodes, weights = np.polynomial.legendre.leggauss(n_u)
    theta = 0.5 * theta_max * (nodes + 1.0)
    w = 0.5 * theta_max * weights
    u = np.tan(theta)
    jac = (1.0 + u * u) / (u * u + 0.25) * w
    return u, jac


DEFAULT_N_U = 2048
DEFAULT_THETA_MAX = np.pi / 2 - 1e-6


def heston_call_prices(
    spot: np.ndarray,
    variance: np.ndarray,
    maturity: float,
    params: HestonParams = PAPER,
    n_u: int = DEFAULT_N_U,
    theta_max: float = DEFAULT_THETA_MAX,
) -> np.ndarray:
    """Lewis 公式，向量化：spot / variance 形状相同（或标量），返回同形状看涨价。

    C = S − √(SK)/π ∫_0^∞ Re[e^{−iuk} φ(u − i/2)] / (u² + 1/4) du，k = log(K/S)。
    """
    s = np.asarray(spot, dtype=float)
    v = np.asarray(variance, dtype=float)
    s, v = np.broadcast_arrays(s, v)
    shape = s.shape
    flat_s, flat_v = s.reshape(-1), v.reshape(-1)
    log_k = np.log(params.strike / flat_s)
    sqrt_sk = np.sqrt(flat_s * params.strike)

    u, jac = _quadrature(n_u, theta_max)
    c_term, d_term = _cf_terms(u - 0.5j, maturity, params)
    cf = np.exp(c_term[None, :] + d_term[None, :] * flat_v[:, None])  # (N, M)
    integrand = np.real(np.exp(-1j * np.outer(log_k, u)) * cf)
    integral = integrand @ jac
    return (flat_s - sqrt_sk / np.pi * integral).reshape(shape)


def heston_call_scalar(
    spot: float, variance: float, maturity: float, params: HestonParams = PAPER
) -> float:
    """自适应积分（quad_vec）的单点参考实现；慢，只用于校验。"""
    log_k = float(np.log(params.strike / spot))

    def integrand(u: np.ndarray) -> np.ndarray:
        uu = np.atleast_1d(u) - 0.5j
        c_term, d_term = _cf_terms(uu, maturity, params)
        cf = np.exp(c_term + d_term * variance)
        return (np.real(np.exp(-1j * np.atleast_1d(u) * log_k) * cf) / (uu.real**2 + 0.25)).ravel()

    integral, _ = quad_vec(integrand, 0.0, np.inf, epsabs=1e-11, epsrel=1e-11, limit=400)
    return float(spot - np.sqrt(spot * params.strike) / np.pi * float(integral[0]))


def varswap_fair_strike(
    variance: np.ndarray, maturity: np.ndarray, params: HestonParams = PAPER
) -> np.ndarray:
    """剩余期限 τ 的方差互换公平方差率（年化方差单位）：E[v_s] = θ + (v−θ)e^{−κs}。

    K(τ) = θ + (v−θ)(1−e^{−κτ})/(κτ)；τ→0 时取极限 v（瞬时方差）。
    """
    v = np.asarray(variance, dtype=float)
    tau = np.asarray(maturity, dtype=float)
    v, tau = np.broadcast_arrays(v, tau)
    tiny = tau < 1e-12
    safe_tau = np.where(tiny, 1.0, tau)
    x = params.kappa * safe_tau
    decay = (1.0 - np.exp(-x)) / x
    strike = params.theta + (v - params.theta) * decay
    return np.where(tiny, v, strike)


def varswap_price_total(
    variance: np.ndarray, time: np.ndarray, maturity: float, params: HestonParams = PAPER
) -> np.ndarray:
    """论文 (5.4) 的 L(t, v)：未来总方差的期望（年化方差 × 年），单位 ≈ 0.0033。

    S²_t = ∫_0^t V_s ds + L(t, V_t) 是论文定义的对冲工具价格（鞅），见 env_heston。
    """
    v = np.asarray(variance, dtype=float)
    t = np.asarray(time, dtype=float)
    v, t = np.broadcast_arrays(v, t)
    tau = np.maximum(maturity - t, 0.0)
    return (v - params.theta) / params.kappa * (1.0 - np.exp(-params.kappa * tau)) + (
        params.theta * tau
    )


def varswap_price_total_dv(
    time: np.ndarray, maturity: float, params: HestonParams = PAPER
) -> np.ndarray:
    """∂L/∂v = (1 − e^{−κ(T−t)})/κ，论文 (5.6) 的 δ2 链式法则因子。"""
    t = np.asarray(time, dtype=float)
    tau = np.maximum(maturity - t, 0.0)
    return (1.0 - np.exp(-params.kappa * tau)) / params.kappa


def bs_call(spot: float, strike: float, maturity: float, sigma: float) -> float:
    """Black–Scholes 看涨价（r = 0）；到期时退化为内在价值。"""
    if maturity <= 0.0 or sigma <= 0.0:
        return float(max(spot - strike, 0.0))
    d1 = (np.log(spot / strike) + 0.5 * sigma**2 * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    return float(spot * norm.cdf(d1) - strike * norm.cdf(d2))


def bs_delta(spot: float, strike: float, maturity: float, sigma: float) -> float:
    if maturity <= 0.0 or sigma <= 0.0:
        return float(spot > strike)
    d1 = (np.log(spot / strike) + 0.5 * sigma**2 * maturity) / (sigma * np.sqrt(maturity))
    return float(norm.cdf(d1))
