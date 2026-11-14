"""一次性诊断 sweep：并行验证 E1 收敛配方。

不改 src/（等胜出配方确定后再回填正式 runner）。每个配方在自己的进程里跑，
单线程、固定 eval 集（所有配方共享同一组 50k 路径，可比），周期性记录 price/δ 误差。

用法：
    python scripts/sweep_recipes.py --list                # 看配方表
    python scripts/sweep_recipes.py --smoke               # 40 iter 冒烟，验证管线
    python scripts/sweep_recipes.py                       # 全量并行
    python scripts/sweep_recipes.py --only a_b1024_cos30k
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 必须在 import torch 之前设置：8 个进程各占一个物理核，多线程只会互相抢核。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")

OUT = ROOT / "results" / "sweep"
BASE_CFG = ROOT / "configs" / "exp" / "e1.yaml"
EVAL_PATHS = 50_000
EVAL_SEED = 987_654
COORD_K = 15

# 论文靶子
TARGET_PRICE = 1.9353
Q_PRICE = 1.691834

JOBS: list[dict[str, Any]] = [
    # --- 诊断：网络到底能不能表示解析 δ？3000 步监督回归，2 分钟出结果 ---
    dict(
        tag="sup_delta",
        note="监督拟合解析δ（诊断可表示性）",
        batch=1024,
        n_iter=3000,
        lr=5e-3,
        lr_min=5e-4,
        schedule="cosine",
        out_scale=[1.0, 100.0],
        warmup=0,
        supervised_only=True,
        eval_every=500,
    ),
    # --- 主配方 A：batch 1024 + cosine ---
    dict(
        tag="a_b1024_cos30k",
        note="Plan A: b1024 cosine 30k",
        batch=1024,
        n_iter=30_000,
        lr=5e-3,
        lr_min=5e-4,
        schedule="cosine",
        out_scale=100.0,
        warmup=0,
        supervised_only=False,
        eval_every=2500,
    ),
    # --- 只差 schedule：恒定 lr，隔离 cosine 的作用 ---
    dict(
        tag="g_b1024_const30k",
        note="b1024 constant 30k（schedule 对照）",
        batch=1024,
        n_iter=30_000,
        lr=5e-3,
        lr_min=5e-3,
        schedule="constant",
        out_scale=100.0,
        warmup=0,
        supervised_only=False,
        eval_every=2500,
    ),
    # --- batch 加大（同样 ~50M 路径预算）---
    dict(
        tag="b_b4096_cos12k",
        note="b4096 cosine 12k",
        batch=4096,
        n_iter=12_000,
        lr=5e-3,
        lr_min=5e-4,
        schedule="cosine",
        out_scale=100.0,
        warmup=0,
        supervised_only=False,
        eval_every=1500,
    ),
    dict(
        tag="e_b8192_cos6k",
        note="b8192 cosine 6k",
        batch=8192,
        n_iter=6_000,
        lr=5e-3,
        lr_min=5e-4,
        schedule="cosine",
        out_scale=100.0,
        warmup=0,
        supervised_only=False,
        eval_every=1000,
    ),
    # --- 小 batch + cosine：隔离 batch 的作用（对照旧配方 256/const/20k=2.77）---
    dict(
        tag="c_b256_cos30k",
        note="b256 cosine 30k（batch 对照）",
        batch=256,
        n_iter=30_000,
        lr=5e-3,
        lr_min=5e-4,
        schedule="cosine",
        out_scale=100.0,
        warmup=0,
        supervised_only=False,
        eval_every=2500,
    ),
    # --- Plan B：解析 δ 热启动 → CVaR ---
    dict(
        tag="d_warm800_cos20k",
        note="warm800 监督 + b1024 cosine 20k（Plan B）",
        batch=1024,
        n_iter=20_000,
        lr=5e-3,
        lr_min=5e-4,
        schedule="cosine",
        out_scale=100.0,
        warmup=800,
        supervised_only=False,
        eval_every=2000,
    ),
    # --- 分腿 out_scale：δ1 腿不再被 100 压缩 ---
    dict(
        tag="f_legscale_cos30k",
        note="b1024 cosine 30k + out_scale[1,100]",
        batch=1024,
        n_iter=30_000,
        lr=5e-3,
        lr_min=5e-4,
        schedule="cosine",
        out_scale=[1.0, 100.0],
        warmup=0,
        supervised_only=False,
        eval_every=2500,
    ),
]


# --------------------------------------------------------------------------- 工具


def lr_at(it: int, total: int, lr0: float, lr1: float, schedule: str) -> float:
    """it 从 1 开始；cosine 从 lr0 降到 lr1。"""
    if schedule != "cosine" or total <= 1:
        return lr0
    frac = (it - 1) / (total - 1)
    return lr1 + 0.5 * (lr0 - lr1) * (1.0 + math.cos(math.pi * frac))


def make_cfg(job: dict[str, Any]) -> dict[str, Any]:
    from src.common.config import load_config

    cfg = load_config(BASE_CFG)
    cfg["batch"] = int(job["batch"])
    cfg["policy"]["out_scale"] = job["out_scale"]
    cfg["policy"]["norm"] = "layer"
    cfg["train"]["n_iter"] = int(job["n_iter"])
    cfg["train"]["lr"] = float(job["lr"])
    return cfg


# --------------------------------------------------------------------------- 评估


def evaluate(
    policy: Any,
    world: Any,
    ev_paths: Any,
    ev_feats: Any,
    bench: Any,
    alpha: float,
) -> dict[str, float]:
    """固定 eval 集上算 price / mean / δ 误差。"""
    import numpy as np
    import torch
    from src.common.metrics import cvar

    policy.eval()
    with torch.no_grad():
        pos = policy.forward_path(ev_feats)
        loss = world.terminal_loss(pos, ev_paths)
        out = {
            "price": float(cvar(loss, alpha)),
            "mean": float(loss.mean()),
            "cv": float(loss.std()),
        }
        spot = ev_paths.spot[:, COORD_K].detach().cpu().numpy()
        var = ev_paths.variance[:, COORD_K].detach().cpu().numpy()
        d1b, d2b = bench.delta(spot, var, COORD_K)
        p1 = pos[:, COORD_K, 0].detach().cpu().numpy()
        p2 = pos[:, COORD_K, 1].detach().cpu().numpy()
        mask = (spot >= 90.0) & (spot <= 110.0)
        if mask.any():
            e1 = p1[mask] - d1b[mask]
            e2 = p2[mask] - d2b[mask]
            out.update(
                {
                    "d1_rmse": float(np.sqrt(np.mean(e1**2))),
                    "d1_mae": float(np.mean(np.abs(e1))),
                    "d1_mean": float(p1[mask].mean()),
                    "d1_bench": float(d1b[mask].mean()),
                    "d2_mae": float(np.mean(np.abs(e2))),
                    "d2_mean": float(p2[mask].mean()),
                    "d2_bench": float(d2b[mask].mean()),
                    "d2_rel_med": float(
                        np.median(np.abs(e2) / np.maximum(np.abs(d2b[mask]), 1e-6))
                    ),
                }
            )
    policy.train()
    return out


# --------------------------------------------------------------------------- 单个配方


def run_job(job: dict[str, Any], n_eval_paths: int = EVAL_PATHS) -> dict[str, Any]:
    import torch
    from src.common.seed import seed_everything
    from src.replication.benchmark import HestonBenchmark
    from src.replication.e1 import market_from_cfg, policy_from_cfg
    from src.replication.env_heston import HestonWorld
    from src.replication.losses import avg_cvar

    torch.set_num_threads(1)
    tag = job["tag"]
    outdir = OUT / tag
    outdir.mkdir(parents=True, exist_ok=True)
    logfile = (outdir / "run.log").open("w", encoding="utf-8", buffering=1)

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(f"[{tag}] {line}", flush=True)
        logfile.write(line + "\n")

    try:
        cfg = make_cfg(job)
        market = market_from_cfg(cfg)
        world = HestonWorld(market, device="cpu")
        alpha = float(cfg["alpha"])
        log(
            f"batch={job['batch']} iters={job['n_iter']} lr={job['lr']}->{job['lr_min']} "
            f"{job['schedule']} out_scale={job['out_scale']} warmup={job['warmup']}"
        )

        t0 = time.perf_counter()
        bench = HestonBenchmark(world)
        log(f"benchmark ready in {time.perf_counter() - t0:.1f}s")

        # 固定 eval 集：所有配方共用同一组路径，price 可直接横向比较
        torch.manual_seed(EVAL_SEED)
        ev_paths = world.simulate(n_eval_paths)
        ev_feats = world.features(ev_paths)
        log(f"eval set {n_eval_paths} paths ready")

        seed_everything(int(job.get("seed", 0)))
        policy = policy_from_cfg(cfg, market.n_features, market.d, torch.device("cpu"))
        opt = torch.optim.Adam(policy.parameters(), lr=float(job["lr"]))

        history: list[dict[str, float]] = []
        supervised_only = bool(job.get("supervised_only"))
        warm = int(job["n_iter"] if supervised_only else job["warmup"])
        warm_time = 0.0

        # ---- 阶段 1：解析 δ 监督热启动（supervised_only 时这就是全部训练）----
        if warm > 0:
            batch = int(job["batch"])
            tw = time.perf_counter()
            for it in range(1, warm + 1):
                paths = world.simulate(batch)
                feats = world.features(paths)
                tgt = bench.positions(paths)
                pos = policy.forward_path(feats)
                w = tgt.pow(2).mean(dim=(0, 1)).sqrt().clamp_min(1e-8)  # 分腿等权
                loss = ((pos - tgt) / w).pow(2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if it % 100 == 0 or it == warm:
                    log(f"  warm {it}/{warm} mse={float(loss.detach()):.4e}")
                if supervised_only and (it % int(job["eval_every"]) == 0 or it == warm):
                    m = evaluate(policy, world, ev_paths, ev_feats, bench, alpha)
                    m["iter"] = it
                    m["elapsed_s"] = time.perf_counter() - tw
                    history.append(m)
                    log(
                        f"  [sup {it}] price={m['price']:.4f} mean={m['mean']:.4f} "
                        f"d1_rmse={m.get('d1_rmse', float('nan')):.4f} "
                        f"d2_mae={m.get('d2_mae', float('nan')):.1f}"
                    )
            warm_time = time.perf_counter() - tw
            log(f"warmup done in {warm_time:.1f}s")
            m = evaluate(policy, world, ev_paths, ev_feats, bench, alpha)
            m["iter"] = 0
            m["warm_s"] = warm_time
            history.append(m)
            log(
                f"  after warm: price={m['price']:.4f} mean={m['mean']:.4f} "
                f"d1_rmse={m.get('d1_rmse', float('nan')):.4f} "
                f"d2_mae={m.get('d2_mae', float('nan')):.1f}"
            )

        # ---- 阶段 2：CVaR 训练 ----
        n_iter = 0 if supervised_only else int(job["n_iter"])
        batch = int(job["batch"])
        eval_every = int(job["eval_every"])
        running = 0.0
        window = 0
        t_train = time.perf_counter()
        for it in range(1, n_iter + 1):
            lr = lr_at(it, n_iter, float(job["lr"]), float(job["lr_min"]), job["schedule"])
            for group in opt.param_groups:
                group["lr"] = lr
            policy.train()
            paths = world.simulate(batch)
            feats = world.features(paths)
            pos = policy.forward_path(feats)
            loss = avg_cvar(world.terminal_loss(pos, paths), alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            window += 1
            if it % 500 == 0 or it == n_iter:
                log(
                    f"  iter {it}/{n_iter} lr={lr:.2e} train_cvar={float(loss.detach()):.4f} "
                    f"avg500={running / window:.4f}"
                )
                running = 0.0
                window = 0
            if it % eval_every == 0 or it == n_iter:
                m = evaluate(policy, world, ev_paths, ev_feats, bench, alpha)
                m["iter"] = it
                m["elapsed_s"] = time.perf_counter() - t_train
                history.append(m)
                log(
                    f"  [eval {it}] price={m['price']:.4f} mean={m['mean']:.4f} "
                    f"d1_rmse={m.get('d1_rmse', float('nan')):.4f} "
                    f"d2_mae={m.get('d2_mae', float('nan')):.1f} "
                    f"({m['elapsed_s'] / 60:.1f}min)"
                )

        total_s = warm_time if supervised_only else warm_time + (time.perf_counter() - t_train)
        final = history[-1] if history else {}
        summary = {
            "tag": tag,
            "note": job["note"],
            "job": {k: v for k, v in job.items() if k != "note"},
            "final": final,
            "best_price": min((h["price"] for h in history), default=float("nan")),
            "history": history,
            "runtime_s": round(total_s, 1),
            "target_price": TARGET_PRICE,
            "q_price": Q_PRICE,
        }
        (outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        torch.save(policy.state_dict(), outdir / "policy.pt")
        log(f"DONE in {total_s / 60:.1f} min  final price={final.get('price', float('nan')):.4f}")
        return summary
    except Exception as exc:  # 一个配方炸了不能带死整轮
        import traceback

        log(f"FAILED: {exc!r}")
        logfile.write(traceback.format_exc() + "\n")
        return {"tag": tag, "note": job.get("note", ""), "error": repr(exc)}
    finally:
        logfile.close()


# --------------------------------------------------------------------------- 汇总


def write_summary(results: list[dict[str, Any]]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        "| tag | note | final price | best price | mean | d1_rmse | d2_mae | wall (min) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for res in results:
        if "error" in res:
            rows.append(f"| {res['tag']} | {res['note']} | FAILED: {res['error']} | | | | | |")
            continue
        f = res.get("final", {})
        rows.append(
            "| {tag} | {note} | {p:.4f} | {bp:.4f} | {m:.4f} | {d1:.4f} | {d2:.0f} | {w:.1f} |".format(
                tag=res["tag"],
                note=res["note"],
                p=f.get("price", float("nan")),
                bp=res.get("best_price", float("nan")),
                m=f.get("mean", float("nan")),
                d1=f.get("d1_rmse", float("nan")),
                d2=f.get("d2_mae", float("nan")),
                w=res.get("runtime_s", 0.0) / 60,
            )
        )
    body = "\n".join(rows)
    text = (
        f"# E1 配方 sweep（靶子 price={TARGET_PRICE}, q={Q_PRICE}）\n\n"
        f"生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}；eval 集 {EVAL_PATHS} 条固定路径，"
        f"δ 误差取第 {COORD_K} 步、S∈[90,110]。\n\n" + body + "\n"
    )
    path = OUT / "summary.md"
    path.write_text(text, encoding="utf-8")
    (OUT / "_runs.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def main() -> int:
    argv = sys.argv[1:]
    if "--list" in argv:
        for job in JOBS:
            print(
                f"{job['tag']:<22} batch={job['batch']:<5} iters={job['n_iter']:<6} "
                f"lr={job['lr']}->{job['lr_min']} {job['schedule']:<8} "
                f"out_scale={job['out_scale']} warmup={job['warmup']}"
            )
        return 0

    selected = JOBS
    if "--only" in argv:
        wanted = argv[argv.index("--only") + 1].split(",")
        selected = [j for j in JOBS if j["tag"] in wanted]
        if not selected:
            print(f"no job matched {wanted}", file=sys.stderr)
            return 2

    smoke = "--smoke" in argv
    if smoke:
        selected = [
            {**j, "n_iter": 40, "eval_every": 20, "warmup": min(int(j["warmup"]), 20)}
            for j in selected
        ]
    n_eval = 1000 if smoke else EVAL_PATHS

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"launching {len(selected)} job(s), eval_paths={n_eval}, smoke={smoke}", flush=True)

    ctx = get_context("spawn")
    procs = []
    for job in selected:
        p = ctx.Process(target=_child, args=(job, n_eval), daemon=False)
        p.start()
        procs.append(p)
        print(f"  started {job['tag']} (pid {p.pid})", flush=True)

    results: list[dict[str, Any]] = []
    for proc, job in zip(procs, selected, strict=True):
        proc.join()
        path = OUT / job["tag"] / "summary.json"
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            results.append({"tag": job["tag"], "note": job["note"], "error": "no summary.json"})

    path = write_summary(results)
    print(f"\nsummary -> {path}", flush=True)
    print(path.read_text(encoding="utf-8"), flush=True)
    return 0


def _child(job: dict[str, Any], n_eval_paths: int) -> None:
    run_job(job, n_eval_paths)


if __name__ == "__main__":
    raise SystemExit(main())
