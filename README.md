# SDSC4107 · Deep Hedging 复现 × 港股交叉验证

[![ci](https://github.com/Casalba/SDSC4107-Group-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Casalba/SDSC4107-Group-project/actions/workflows/ci.yml)

> 状态：**A1 完成**（骨架 + pfhedge 适配层 + 检查点 B0 + E1 五 seed + Fig 2/3/4 + 英文方法说明）。
> 下一步：A2（E2/E3/E4）与 B 组（港股、回测，接入 `policy_heston_d1`）。

## 先读两份文档

| 文件 | 管什么 |
|---|---|
| [docs/plan.md](docs/plan.md) | 做什么、做到什么算对：E1–E4 复现靶子、港股 C\* 口径、判据与风险 |
| [docs/spec.md](docs/spec.md) | 谁做什么、交到哪、怎么命名、怎么验收：角色边界、目录命名、接口 schema、每人 DoD、Gate |

冲突时：范围与判据看 plan，格式与流程看 spec。

A1 的方法与结果汇总（中文，含数字表、图与复现命令）：[docs/report_a1.md](docs/report_a1.md)。

## 目录

```
project/
├─ README.md            # 本文件
├─ pyproject.toml       # pytest / ruff 配置
├─ requirements.lock.txt
├─ .venv/               # Python 3.12 环境（约 1.0 GB，已从上级目录迁入）
├─ docs/
│   ├─ plan.md          # 方案 v1.4
│   ├─ spec.md          # 协作规范 v1.2
│   ├─ experiments.md   # 实验登记表（spec §9）
│   ├─ b0.md            # 检查点 B0：三条对拍与通过与否
│   ├─ e1_diagnosis.md  # E1 收敛问题的诊断、8 个配方、5 seed 表（§6 定稿）
│   ├─ report_a1.md     # A1 报告：方法 + 结果 + 复现命令
│   ├─ changelog.md     # 文档/结果同步的变更记录与待办
│   ├─ method_replication.md  # 英文方法说明（§5 的复现口径与 deviation 表）
│   ├─ paper/           # deephedging.pdf + section5.txt / section5b.txt（论文与摘录）
│   └─ paper_fulltext_v1.txt  # 机器抽取全文（91 KB，顶部有 LOSSY 警告；仅供检索）
├─ src/
│   ├─ common/          # config seed logging metrics types
│   ├─ policy/          # policy.py（冻结接口）+ weights/heston_d1.pt（A1→B2 交接件）
│   └─ replication/     # env_heston env_bs net_recurrent losses benchmark pricing e1 train smoke
├─ configs/exp/         # debug.yaml / e1.yaml / e1d1.yaml
├─ tests/
├─ scripts/             # ci / check_b0 / make_figures / sweep_recipes / export_policy_d1 / recompute_d1_delta
│   └─ diagnostics/     # 一次性诊断探针（追溯用，不参与 CI）
├─ figures/             # fig02_loss_hist / fig03_delta_surface / fig04_recurrent_vs_simpler
├─ reports/             # 英文正文（A2 起建；A1 的正文在 docs/method_replication.md）
└─ refs/                # 第三方仓库快照，只读参考，不改、不提交
    ├─ pfhedge/         # pfnet-research/pfhedge @ 1fc08c7 (2024-08-30)
    └─ deephedging/     # hansbuehler/deephedging @ fe53fd9 (2026-01-22)（notebooks 保留）
```

`data/`、`results/` 运行时生成、不进 git（spec §4 / §7）。

## 跑起来（5 行）

```powershell
cd E:\cityu\sdsc4107\project
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt   # 首次装环境
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m src.replication.train --config configs/exp/debug.yaml
# -> results/debug/0/{metrics.json, config.yaml, run.log, pnl.npz}
```

本地 CI（ruff + pytest，与 GitHub Actions 同源）：`.venv\Scripts\python.exe scripts\ci.py`。

## A1 现在能跑什么

下面每条都用 `.venv\Scripts\python.exe -m ...` 从仓库根目录跑。

| 命令 | 干什么 | 产物 |
|---|---|---|
| `scripts.check_b0` | Heston 价 / 方差互换 / 中间时点三条对拍（4 s） | `results/b0/b0.json`，结论在 [docs/b0.md](docs/b0.md) |
| `src.replication.train --config configs/exp/e1.yaml --seed k` | E1 一个 seed（2 线程约 32 min，5 个并行更划算） | `results/e1/k/`（含 `metrics.json`） |
| `scripts.make_figures` | Fig 2/3/4，需要 `results/e1/0..4` | `figures/*.png`（300 dpi） |
| `scripts.export_policy_d1` | 冻结 d=1 策略，交 B2 回测 | `src/policy/weights/heston_d1.pt` |
| `scripts.sweep_recipes --list` | 8 个训练配方的并行对照（定了 E1 的配方） | `results/sweep/summary.md` |

E1 定稿数字：5 seed price **1.9431 ± 0.0007**（论文 1.94，解析 δ 对冲 1.9346），
mean loss 1.6913 vs q 1.69183。配方怎么选出来的见 [docs/e1_diagnosis.md](docs/e1_diagnosis.md) §6。

## 参考仓库怎么看

- `refs/pfhedge/`：我们要用的库本体。看 `pfhedge/nn/modules/`（风险测度）、`pfhedge/instruments/`（HestonStock / VarianceSwap）、`examples/`（训练循环写法）。本地已装 0.23.0 于 `project/.venv`。
- `refs/deephedging/`：论文原作者的 TensorFlow 实现，看 `objectives.py`（风险测度）、`layers.py`（网络结构）、`world.py`（Heston 世界）。68 MB 里大部分是 `notebooks/*.ipynb`，源码只有 ~10 个 `.py`。

## 环境

- Python 环境就在本目录：`project\.venv`（约 1.0 GB，已从上级 `.venv` 迁入，console 脚本已重生成）。
  - 直接调用：`.venv\Scripts\python.exe -m pip ...`；`pip` / `pip3` / `pip3.12` 已修复可用。
  - 已装（完整清单见 `requirements.lock.txt`）：pfhedge 0.23.0、torch 2.15.0+cpu、numpy 1.26.4、scipy 1.13.1、pandas 3.0.6、matplotlib 3.11.2、sympy 1.14.0、PyYAML 6.0.3、pytest 9.2.0、ruff 0.17.5。
  - **numpy 必须 <2**：pfhedge 0.23.0 声明 `numpy<2.0,>=1.26`；scipy 相应钉 1.13.1（scipy ≥1.18 要求 numpy≥2，会互相打架）。装包时先 `numpy==1.26.4` 再 `scipy==1.13.1`，最后 `pip check` 确认无冲突。
  - 装包加 `-i https://pypi.org/simple`：本机默认清华源对部分 wheel 返回 HTTP 403。
- **torch 目前是 CPU 版，这是刻意的**：组里四台机器不统一，代码用 `device=auto`，CPU 也能跑通全流程。有 N 卡的（我这台 5070 Ti 属 sm_120，要 cu128 版）各自按机器装；每个结果里的 `device` / `torch_version` 会记录是哪台机器跑的（spec §5.2）。
- 数据不进 git：`data/`、`results/` 都忽略，抓取脚本在 `scripts/`（开工后建）。
