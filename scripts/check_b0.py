"""B0 对拍（plan §2.5）：定价器 / 方差互换闭式 / 中间时点 mark。

三条全过（阈值见 plan §2.5）才允许 E1 的直方图与 delta 面对比走"解析 + 闭式"路线；
否则只砍这两块，E2–E4 不受影响。

用法：python -m scripts.check_b0        （约 1–2 分钟）
输出：results/b0/b0.json（含全部数值与 passed 标志）
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from pfhedge.stochastic import generate_heston
from src.common.seed import seed_everything
from src.replication.pricing import (
    PAPER,
    heston_call_prices,
    varswap_price_total,
)

DT = PAPER.maturity / 30.0
N_STEPS = 30
OUT = Path("results/b0/b0.json")


def check_call(n_paths: int = 200_000) -> dict[str, float | bool | str]:
    """① Heston 解析 call vs pfhedge QE-M MC。"""
    seed_everything(0)
    spot, _ = generate_heston(
        n_paths,
        N_STEPS + 1,
        init_state=(PAPER.s0, PAPER.v0),
        kappa=PAPER.kappa,
        theta=PAPER.theta,
        sigma=PAPER.sigma,
        rho=PAPER.rho,
        dt=DT,
        dtype=torch.float64,
    )
    payoff = torch.clamp(spot[:, -1] - PAPER.strike, min=0.0).numpy()
    mc = float(payoff.mean())
    se = float(payoff.std(ddof=1) / np.sqrt(n_paths))
    analytic = float(
        heston_call_prices(np.array([PAPER.s0]), np.array([PAPER.v0]), PAPER.maturity)[0]
    )
    rel = abs(mc - analytic) / analytic
    return {
        "name": "heston_call_analytic_vs_mc",
        "analytic": analytic,
        "mc": mc,
        "mc_se": se,
        "rel_err": rel,
        "threshold": 0.015,
        "passed": bool(rel < 0.015),
    }


def check_varswap(n_paths: int = 200_000) -> dict[str, float | bool | str]:
    """② 方差互换：鞅性 + 闭式（bT）vs MC 的 E[∫V ds]。

    ∫V ds 按步内条件期望累加（与 env_heston 同一口径）：
    E[∫Δ V ds | V_j] = (V_j−b)(1−e^{−καΔ})/α + bΔ。用左矩形会在状态相关项上引入
    −(V_j−b)αΔ²/2 的系统偏差，使 S² 不再是鞅（网络会去套这个漂移）。
    """
    seed_everything(1)
    _, var = generate_heston(
        n_paths,
        N_STEPS + 1,
        init_state=(PAPER.s0, PAPER.v0),
        kappa=PAPER.kappa,
        theta=PAPER.theta,
        sigma=PAPER.sigma,
        rho=PAPER.rho,
        dt=DT,
        dtype=torch.float64,
    )
    v = var[:, :-1].numpy()
    accrual = (v - PAPER.theta) * (-np.expm1(-PAPER.kappa * DT) / PAPER.kappa) + PAPER.theta * DT
    integral = accrual.sum(axis=1)
    mc = float(integral.mean())
    se = float(integral.std(ddof=1) / np.sqrt(n_paths))
    closed = float(varswap_price_total(PAPER.v0, 0.0, PAPER.maturity))
    rel = abs(mc - closed) / closed
    return {
        "name": "varswap_martingale_vs_closed_form",
        "closed_form": closed,
        "mc": mc,
        "mc_se": se,
        "rel_err": rel,
        "threshold": 0.01,
        "passed": bool(rel < 0.01),
    }


def check_mid_mark(
    n_states: int = 1000, reps: int = 300, k_mid: int = 15
) -> dict[str, float | bool | str]:
    """③ 中间时点 mark：闭式 C(S_t, V_t, τ) vs 从该状态重模拟的条件 MC。

    用方差积分做控制变量（其条件期望 L(t, V_t) 已知），否则 300k 条路径的
    se ≈ 1%，不足以在 1% 阈值上分辨。同时报告方差 mark 的偏差（应为 0）。
    """
    seed_everything(2)
    n_paths = 100_000
    spot, var = generate_heston(
        n_paths,
        N_STEPS + 1,
        init_state=(PAPER.s0, PAPER.v0),
        kappa=PAPER.kappa,
        theta=PAPER.theta,
        sigma=PAPER.sigma,
        rho=PAPER.rho,
        dt=DT,
        dtype=torch.float64,
    )
    idx = torch.randperm(n_paths)[:n_states]
    s_mid = spot[idx, k_mid]
    v_mid = var[idx, k_mid]
    taus = PAPER.maturity - k_mid * DT

    s_rep = s_mid.repeat_interleave(reps)
    v_rep = v_mid.repeat_interleave(reps)
    seed_everything(3)
    spot_rest, var_rest = generate_heston(
        n_states * reps,
        N_STEPS + 1 - k_mid,
        init_state=(s_rep, v_rep),
        kappa=PAPER.kappa,
        theta=PAPER.theta,
        sigma=PAPER.sigma,
        rho=PAPER.rho,
        dt=DT,
        dtype=torch.float64,
    )
    payoff = torch.clamp(spot_rest[:, -1] - PAPER.strike, min=0.0).numpy().reshape(n_states, reps)
    integ = ((var_rest[:, :-1]) * DT).sum(dim=1).numpy().reshape(n_states, reps)

    known_integ = np.array(
        [varswap_price_total(float(v), k_mid * DT, PAPER.maturity) for v in v_mid.numpy()]
    )
    closed = heston_call_prices(s_mid.numpy(), v_mid.numpy(), taus)

    # 控制变量：逐状态中心化后回归（用池化 β）
    xc = (integ - known_integ[:, None]).ravel()
    yc = (payoff - payoff.mean(axis=1, keepdims=True)).ravel()
    beta = float(np.dot(xc, yc) / np.dot(xc, xc))
    y_cv = payoff.mean(axis=1) - beta * (integ.mean(axis=1) - known_integ)
    diff = y_cv - closed
    se = float(np.std(diff, ddof=1) / np.sqrt(n_states))
    rel = float(abs(diff.mean()) / closed.mean())
    var_rel = float(abs((integ.mean(axis=1) - known_integ).mean()) / known_integ.mean())
    return {
        "name": "mid_time_mark_vs_conditional_mc",
        "closed_form": float(closed.mean()),
        "mc_cv": float(y_cv.mean()),
        "rel_err": rel,
        "se": se,
        "cv_beta": beta,
        "varswap_mark_rel_err": var_rel,
        "threshold": 0.01,
        "passed": bool(rel < 0.01),
    }


def main() -> int:
    t0 = time.time()
    results = [check_call(), check_varswap(), check_mid_mark()]
    payload = {
        "schema_version": "1.0",
        "params": {
            "kappa": PAPER.kappa,
            "theta": PAPER.theta,
            "sigma": PAPER.sigma,
            "rho": PAPER.rho,
            "v0": PAPER.v0,
            "s0": PAPER.s0,
            "strike": PAPER.strike,
            "maturity": PAPER.maturity,
            "n_steps": N_STEPS,
            "dt": DT,
        },
        "checks": results,
        "passed": all(bool(r["passed"]) for r in results),
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    for r in results:
        flag = "PASS" if r["passed"] else "FAIL"
        print(f"[{flag}] {r['name']}: rel_err={r['rel_err']:.4%} (threshold {r['threshold']:.1%})")
    print(f"passed={payload['passed']} runtime={payload['runtime_sec']}s -> {OUT}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
