# Deep Hedging 项目 · 协作规范 v1.2

> 配套文件：`docs/plan.md`（v1.3，定义**做什么、做到什么算对**）。本文件定义**谁做、交什么、放哪、怎么命名、怎么验收**。
> 冲突时：**范围与判据以 plan.md 为准，格式与流程以本文件为准**。
> 本文件改动需全员知会，版本号 +1，并在 §12 记录。

---

## 0. 一页速查

| 人 | 一句话职责 | 一句话交付 | 硬依赖 |
|---|---|---|---|
| A1 | 骨架 + Heston d=2 基准 + E1 | 可复跑的复现流水线 + `policy_heston_d1` | 无（B2-M1 等它） |
| A2 | 风险偏好 + 成本 + 定价渐近（E2–E4） | 四张图 + 两条斜率 | A1 的适配层 |
| B1 | 港股数据 + 校准 + 路径生成器 | `data/processed/hk_daily.csv` + 校准参数 | 无 |
| B2 | 回测 + C\* + 统计 | 港股主表 + 结论段 | A1 的 d=1 policy、B1 的数据 |

**四条铁律**
1. **结果只能由脚本产生**。`results/` 下的任何文件都必须能由一条命令重跑；手改结果一律视为作弊，需要重跑。
2. **先过检查点再往下走**：B0（对拍）不过，不开始 E1 训练。
3. **接口先冻结再并行**：`src/policy/policy.py`、`src/common/metrics.py` 的签名一旦合并，改动要 PR + 两人 review。
4. **每人每周至少一次 push 到自己的分支**，不留大块未提交的本地改动。

---

## 1. 角色边界（谁不碰什么）

越界是这类项目最常见的返工来源，先划清：

| 路径 | 归属 | 其他人 |
|---|---|---|
| `src/common/` | A1（B 组可提 PR） | 只读，改动需 A1 review |
| `src/replication/` | A1、A2 | B 组只读 |
| `src/policy/` | A1（接口）、A2（d=2 变体） | B 组只调用，不改 |
| `src/hk/` | B1（data/calibrate/synth）、B2（backtest/stats） | A 组只读 |
| `data/` | B1 | 只读 |
| `configs/exp/` | A1、A2 | — |
| `configs/hk/` | B1、B2 | — |
| `results/` | 各自脚本写入 | 不手改、不互相覆盖 |
| `docs/`、`reports/`、`figures/` | 全员写各自小节 | 改动他人小节需知会 |

**禁止**：
- 把 notebook 当交付物。notebook 只做探索，结论要落成 `src/` 里的 `.py` + 一条命令。
- 提交 `data/`、`.venv/`、`results/` 里的大文件、模型权重（见 §7 `.gitignore`）。
- 用未来数据：港股必须严格 walk-forward，任何"全样本校准后回测"的写法都要在 PR 里标红。
- 私自改字段名（`cvar95` 不能写成 `CVaR_95`）。

---

## 2. 仓库与目录规范

```
project/                     # = E:\cityu\sdsc4107\project，课程材料留在上级目录
├─ README.md                 # 5 行跑法 + 环境安装 + 目录说明
├─ pyproject.toml            # ruff / pytest 配置
├─ requirements.lock.txt     # pip freeze，禁止只写 requirements.txt
├─ .venv/                    # Python 3.12 环境（已从上级迁入；gitignored）
├─ .gitignore
├─ docs/
│   ├─ plan.md               # 范围与判据（权威）
│   ├─ spec.md               # 本文件
│   ├─ paper/                # deephedging.pdf + section5.txt / section5b.txt
│   ├─ data.md               # 数据来源/字段/清洗/缺口（B1 维护）
│   ├─ experiments.md        # 实验登记表（全员追加，见 §9）
│   ├─ method_replication.md # 复现方法说明（A1，英文）
│   └─ report/               # 报告草稿，英文
├─ refs/                     # 第三方仓库快照：只读参考，不修改、不提交
│   ├─ pfhedge/              # pfnet-research/pfhedge @ 1fc08c7 (2024-08-30)
│   └─ deephedging/          # hansbuehler/deephedging @ fe53fd9 (2026-01-22)
├─ src/
│   ├─ common/               # config.py seed.py logging.py metrics.py types.py
│   ├─ replication/          # env_heston.py env_bs.py net_recurrent.py losses.py
│   │                        # benchmark.py train.py evaluate.py
│   ├─ hk/                   # data.py calibrate.py synth.py backtest.py stats.py
│   └─ policy/               # policy.py（冻结接口）
├─ configs/
│   ├─ exp/                  # e1.yaml e2.yaml e2cs.yaml e3.yaml e4bs.yaml e4h.yaml e4low.yaml
│   └─ hk/                   # fut1bp.yaml fut10bp.yaml etf30bp.yaml
├─ data/                     # gitignored，只留 README + checksum
│   ├─ raw/<source>/<name>_<YYYYMMDD>.csv
│   ├─ interim/
│   └─ processed/hk_daily.csv
├─ results/                  # gitignored，结构见 §4
├─ figures/                  # 报告终版图（从 results 复制并改名）
├─ reports/                  # 英文正文
├─ scripts/                  # fetch_hk.py check_b0.py make_figures.py repro.py
└─ tests/
```

**硬规则**
- 新目录必须先在 PR 里说明用途，再提交（避免 `tmp/`、`test2/` 这类目录）。
- 任何 `.py` 文件不超过 600 行；超了就拆。
- 所有路径用 `pathlib.Path`，不写死绝对路径（`E:\cityu\...` 只能出现在 README 示例里）。

---

## 3. 命名规范

### 3.1 实验 id（白名单）

| exp id | 含义 | seed | 产出目录 |
|---|---|---|---|
| `e1` | 零成本 α=0.5，Heston d=2 | 0–4 | `results/e1/<seed>/` |
| `e2` | 零成本 α=0.99 | 0–4 | `results/e2/<seed>/` |
| `e2cs` | call spread（K1=100,K2=101）α∈{0.95,0.99} | 0–4 | `results/e2cs/<alpha>/<seed>/` |
| `e3` | ε=0.01，recurrent vs simpler | 0–4 | `results/e3/<seed>/` |
| `e4bs` | 渐近，BS d=1 | 0–4 | `results/e4bs/<eps>/<seed>/` |
| `e4h` | 渐近，Heston d=2 | 0–4 | `results/e4h/<eps>/<seed>/` |
| `e4low` | 低摩擦附表 ε∈{0.1,0.03,0.01} | 0–4（可 0–2，需在 log 注明） | `results/e4low/<eps>/<seed>/` |
| `e1d1` | d=1 版本，供 B 组迁移（硬接口） | 0–4 | `results/e1d1/<seed>/` |
| `debug` | 骨架冒烟，只验证流水线能跑 | 0 | `results/debug/0/` |
| `hk` | 港股回测 | — | `results/hk/<cost_tag>/<policy_tag>/` |

港股 tag：`cost_tag ∈ {fut1bp, fut10bp, etf30bp}`，`policy_tag ∈ {bench, m1, m2}`。
B1 的校准参数另落 `results/calib/<index>_garch.json`，不占 exp 目录。
**不在白名单里的 exp id 不允许建目录**（要加先改本文件）。

### 3.2 文件命名

| 类型 | 规则 | 例 |
|---|---|---|
| 配置 | `configs/<组>/<exp_id>.yaml` | `configs/exp/e1.yaml` |
| 指标 | `metrics.json`（固定 schema，见 §5.2） | `results/e1/0/metrics.json` |
| 路径数据 | `pnl.npz`（键名固定，见 §5.3） | `results/e1/0/pnl.npz` |
| 图 | `fig_<论文图号>_<slug>.png` | `fig_02_loss_hist_alpha05.png` |
| 日志 | `run.log`（stdout+stderr 原样落盘） | `results/e1/0/run.log` |
| 终端图 | `figures/fig<论文图号>_<slug>.png` | `figures/fig02_loss_hist.png` |
| 数据 | `<name>_<YYYYMMDD>.csv`，日期=抓取日 | `data/raw/yahoo/hsi_20260901.csv` |
| 报告 | `reports/<section>_<slug>.md` | `reports/e1_results.md` |

命名只用小写英文 + 数字 + 下划线；**中文只出现在 `docs/` 的正文里**。

### 3.3 Git

- 分支：`feat/<role>-<slug>`、`exp/<exp_id>`、`data/<slug>`、`fix/<slug>`，例 `feat/a1-benchmark`、`exp/e2`。
- 提交：`<type>(<scope>): <一句话>`，type ∈ feat/fix/exp/data/docs/chore，例 `feat(benchmark): heston call via carr-madan`。
- **不直接 push main**；PR 至少 1 人 review（跨组改接口要 2 人）；CI 绿才可合。

---

## 4. 结果目录规范

```
results/<exp>/<...>/<seed>/
├─ metrics.json     # 必需，§5.2 schema
├─ config.yaml      # 必需，本次运行的实际配置快照（含所有默认值）
├─ run.log          # 必需
├─ pnl.npz          # 必需（E1–E4、港股都存终端损失）
└─ fig_*.png        # 该 seed 的图（可选，终版在 figures/）
```

- `config.yaml` 是**运行时的实际配置**（argparse 默认值合并后），不是原始文件拷贝。
- `metrics.json` 里必须带 `git_commit`（运行时的 `git rev-parse --short HEAD`）；工作区脏（`git status --porcelain` 非空）时加 `"dirty": true`，这类结果的数字不许进报告。
- `results/` 不进 git，但**每周备份一次**（外接盘或私有网盘），备份前跑 `python -m scripts.repro --check`。

---

## 5. 接口规范（冻结）

### 5.1 策略接口

```python
# src/policy/policy.py
class Policy(Protocol):
    def reset(self) -> None: ...                 # 每个 episode（一条路径批次）开头调用
    def forward(self, x_k: Tensor) -> Tensor: ...  # x_k: (B, f) -> position: (B, d)
```

- 输入特征（**v1.2 冻结**）：d=1 时 `x_k = [log(S_k/S_0), t_k/T]`（f=2）；d=2 时 `x_k = [log(S_k/S_0), t_k/T, varswap_k/varswap_0]`（f=3）；书面形状 `(B, f)`。
  - `varswap_k` = 论文式 (5.4) 的方差互换价格 `S²_k = E_{t_k}[∫_0^T V_s ds]`（剩余段积分 + `L(t_k,V)`；可交易 MTM 价格、martingale；`S²_0 = bT ≈ 0.0033`）。**不是**年化 fair strike（`S²/T`，量级 0.04，带漂移、不可作输入）、**不是**瞬时方差 `v_k`；`varswap_0` 同口径。`k = n` 无交易，不需要该点。（v1.4 修正：0.04 是年化口径，价格口径是 0.0033）
  - 网络宽度：输入层 `f + d`（d=1 为 3、d=2 为 5），隐层 `N1 = N2 = d+15`，输出 `d`。与论文 `N0 = 2d`、`I_k = (log S¹_k, V_k)` 的差异是 documented deviation，写进 `docs/method_replication.md`。
- `position` 的单位 = **标的份额**（论文 δ 的口径），即 `δ_i` 份工具 i；成本按**名义本金** `c_k = Σ_i ε_i |δ_i,k − δ_i,k−1| S_i,k` 计，首次建仓 `δ_i,−1 = 0` 计入、到期不强制平仓（论文 (5.7)）。
  - `S_i,k` 是**工具价格**：标的 ≈100、方差互换 ≈0.0033（`bT`） → 同一 ε 下标的腿单位成本约为互换腿的 **≈3×10^4 倍**。E3 的解释必须按这个口径（见 plan §10.5）。
- `reset()` 必须清空 recurrent 隐状态。忘了写 `reset` 是 A 组最常见的 bug，单测要覆盖。
- **`policy_heston_d1`**：A1 交付，`forward` 输入 f=2、输出 d=1，B 组直接 import 使用，不得改。

### 5.2 `metrics.json` schema

固定外壳 + 按 exp 的 `metrics` 子对象：

```json
{
  "schema_version": "1.0",
  "exp": "e1",
  "seed": 0,
  "git_commit": "c3bc2ab",
  "dirty": false,
  "created_at": "2026-09-01T10:00:00+08:00",
  "runtime_sec": 1234.5,
  "device": "cpu",
  "torch_version": "2.15.0+cpu",
  "n_paths": 100000,
  "config_sha256": "…",
  "metrics": { },
  "checks": { "pth0_in_range": true, "delta_rmse_ok": true },
  "passed": true
}
```

`metrics` 子对象的必填键：

| exp | 必填键 |
|---|---|
| e1 | `price`, `q`, `mean_loss`, `cvar50`, `delta_rmse`, `price_simpler` |
| e2 | `price`, `cvar05_own`, `cvar99_own`, `cvar05_other`, `cvar99_other` |
| e2cs | `band_width_95`, `band_width_99` |
| e3 | `price_recurrent`, `price_simpler`, `cvar_recurrent`, `cvar_simpler`, `mean_loss_recurrent`, `mean_loss_simpler` |
| e4bs / e4h / e4low | `eps`, `price_eps`, `price0`, `slope`, `slope_se`, `r2` |
| e1d1 | `price`, `mean_loss`, `cvar50`（口径同 e1，只是 d=1） |
| hk | `c95`, `c99`, `mean_loss`, `turnover`, `cost_notional`, `n_windows` |

- 数值精度：报告里 3 位有效数字；`metrics.json` 存原始 float。
- `device` 与 `torch_version` 必填。四个人机器不一样（有人只有 CPU，有人 4060 / 5070 Ti），同一实验跨 device 会有 1e-3 级差别：**报告里一组数字必须来自同一 device**，跨人对比前先看这两个字段。
- 单位写进键名或 `checks`，不靠注释：`cost_notional` 是"成本 / 名义本金"。

### 5.3 `pnl.npz` 键名

| 键 | 形状 | 含义 |
|---|---|---|
| `terminal_loss` | `(n_paths,)` | 终端损失 `L = Z − (δ·S)_T + 成本`（C\* 的输入） |
| `wealth_path` | `(n_paths, n+1)` | 可选，财富过程 |
| `positions` | `(n_paths, n, d)` | 可选，仅调试时存（大） |

### 5.4 共享函数（`src/common/metrics.py`，A1 写、双方用）

```python
def cvar(x: Tensor, alpha: float) -> Tensor: ...
def capital_requirement(loss: Tensor) -> dict[str, float]:  # {"c95":…, "c99":…}
def turnover(positions: Tensor) -> Tensor: ...
def proportional_cost(positions: Tensor, prices: Tensor, eps: float) -> Tensor: ...
```

- `capital_requirement` 必须是**闭式**（`C* = CVaR(loss)`，题目已证现金可加），不许在里面做求根/优化；有单测对齐手算。
- 港股与复现用**同一份**实现，出现两套算法即视为 bug。

---

## 6. 每人任务与交付标准（DoD）

> 通则：每项任务的"完成"= 代码合入 + 结果可复跑 + `docs/experiments.md` 追加一行 + 一句话结论（英文，≤30 词）。

### A1 —— 骨架 + 基准 + E1

| # | 任务 | 交付物 | DoD（可勾选） |
|---|---|---|---|
| 1 | 仓库骨架 | `src/common/*`, `pyproject.toml`, `tests/`, CI 配置 | `pytest -q` 绿；`python -m src.replication.train --config configs/exp/debug.yaml` 能跑出 `results/debug/0/metrics.json`；README 有 5 行跑法 |
| 2 | pfhedge 适配层 + BS 环境 | `env_heston.py`, `env_bs.py`, `net_recurrent.py`, `losses.py` | 单测：两套环境路径形状 `(B,31)`、成本为正、avg-CVaR 与手算一致、BS 解析价 vs MC 差 <1%；`reset()` 后隐状态归零 |
| 3 | benchmark + 检查点 B0 | `benchmark.py`, `scripts/check_b0.py`, `docs/` 里一页 B0 记录 | 三个对拍数值 + 通过/不通过写清楚；不通过要写明走了哪条降级路径 |
| 4 | E1 训练 + 5 seed + **d=1 版本** | `results/e1/0..4/`, `results/e1d1/0..4/` | 5 个 seed 全出；`policy_heston_d1` 权重与加载脚本交给 B 组并当面跑通一次 |
| 5 | E1 的 Fig2/3/4 | `figures/fig02_*`, `fig03_*`, `fig04_*` | 300 dpi、英文轴标签、caption 一句话结论、图上标注论文参考值 |
| 6 | 方法说明 | `docs/method_replication.md`（英文，≥2 页） | 覆盖：设定、网络、loss、benchmark 实现、与论文的差异点 |

### A2 —— 风险偏好 + 成本 + 渐近

| # | 任务 | 交付物 | DoD |
|---|---|---|---|
| 1 | E2（α=0.99） | `results/e2/0..4/`, `fig06_*`, `fig07_*` | 两条损失分布**交叉**（0.527 vs 0.253、1.803 vs 2.363 同向）；不交叉要写排查记录 |
| 2 | E2 的 call spread | `results/e2cs/<alpha>/0..4/`, `fig08_*`, `fig09_*` | 定性图 + 仓位区间宽度表；α=0.99 明显收窄 |
| 3 | E3（ε=0.01） | `results/e3/0..4/`, 三行表 | recurrent 5.51 / simpler 6.74（±10%）且 recurrent 严格更优 |
| 4 | E4-BS | `results/e4bs/<eps>/<seed>/`, `fig10_*` | 斜率 0.67±0.04、R²>0.95 |
| 5 | E4-Heston + 低摩擦附表 | `results/e4h/...`, `results/e4low/...`（先 seed 0–2）, `fig11_*` | 斜率 0.71±0.05；低 ε 段给出"能分辨/无法分辨"的明确结论 |
| 6 | 结论段草稿 | `reports/e2_e4_findings.md`（英文） | 每张图一句结论 + 与论文数值的对照表 |

> 前置：E4-BS 用 A1 的 `env_bs.py` + entropic loss；E4-Heston 的 `p0` 用 A1 的 Carr–Madan 定价器（G1 之后）。E2/E3 不依赖 benchmark，G0 一过即可开跑。

### B1 —— 数据 + 校准 + 生成器

| # | 任务 | 交付物 | DoD |
|---|---|---|---|
| 1 | `^VHSI` 侦察 | `docs/data.md` 一节 | 明确"可得/不可得"，可得则给出抓取脚本 + 覆盖区间 |
| 2 | 抓取 5 序列 | `scripts/fetch_hk.py`（可复跑）, `data/raw/…` | 重跑后 checksum 一致（或差异有解释）；抓取日期写进文件名 |
| 3 | 清洗与对齐 | `data/processed/hk_daily.csv` | 列固定：`date, hsi, hsce, etf2800, div_yield, hibor1m, vhsi`；缺失率与处理方式写进 `data.md` |
| 4 | 数据体检 | `docs/data.md` 的体检节 + 图 | stylized facts（波动聚集、尖峰厚尾）、跳跃检验、regime 划分（2015H2/2020H1/2022）各一句结论 |
| 5 | GARCH 校准 | `results/calib/<index>_garch.json` | 参数 + 标准误 + 与历史波动率对照图；残差无显著自相关 |
| 6 | 路径生成器 | `src/hk/synth.py` + 单测 | 生成路径的均值/方差/自相关与校准参数在 5% 内一致；接口与 A 组模拟器对齐 |
| 7 | 数据文档 + 联调 | `docs/data.md` 完整版 | 来源、字段、假设、缺口、许可证（课程内使用、不再分发）齐全 |

### B2 —— 回测 + C\* + 统计

| # | 任务 | 交付物 | DoD |
|---|---|---|---|
| 1 | 回测引擎 | `src/hk/backtest.py` + 单测 | 手算 1 个窗口核对：仓位、结算、成本、整手取整逐项对上 |
| 2 | 指标 + bootstrap | `src/hk/stats.py` | `capital_requirement` 复用 `common/metrics.py`；block bootstrap（block=12，B=2000）有单测（已知分布上 CI 覆盖率合理） |
| 3 | Bench 腿 | `results/hk/<cost>/bench/` | delta 调仓 + 成本正确；与解析 BS delta 在无成本时差 < 1e-6 |
| 4 | M1 / M2 | `results/hk/<cost>/m1/`, `m2/` | M1 直接吃 A1 的 d=1 policy；M2 用 B1 的 GARCH 路径重训，训练/测试严格分开 |
| 5 | 主表 + 敏感性 | `results/hk/…/metrics.json` 齐, 表 + 图 | 两档成本 × 三条策略 × {C\*95,C\*99} 齐；P1–P4 各有明确答案 |
| 6 | 结论段 | `reports/hk_findings.md`（英文） | 与 §4.1 两个预期逐条对账，打脸也照写 |

---

## 7. 数据与合规

- 数据只能用于本课程作业；**不公开再分发**，仓库里只放 `scripts/fetch_hk.py`，不放原始 csv。
- 每次抓取在 `docs/data.md` 记：来源 URL、抓取日期、字段名、行数、checksum。
- `.gitignore` 至少包含：`data/`, `results/`, `*.npz`, `*.pt`, `*.pth`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`。
- 交易日历用 HKEX 官方日历（节假日对齐），不要用 pandas 的 business day 近似；时区统一 `Asia/Hong_Kong`。
- 缺失处理只有两种：前值填充（并置 `is_filled=1` 标记）或整窗剔除。不许插值后当真实价格。

---

## 8. 代码规范

- Python ≥3.10，venv 固定在 `project/.venv`（已从上级迁入，console scripts 已重生成）。**torch 版本不强制统一**：四人机器不同，各人按自己机器装（CPU 版即可跑通全流程；有 N 卡的装对应 cu 版本，5070 Ti 是 sm_120 → cu128），版本写进 `requirements.lock.txt` 并在 `metrics.json` 里记录。
- 设备选择只用一行，写死设备是 review 会拒的写法：

  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```

  config 里 `device: auto`；代码里不许出现裸 `.cuda()`，也不许假设多卡。
- 每个函数带类型标注；公共函数必须有 docstring（一行说清输入输出单位）。
- `ruff check` + `ruff format` 过 CI；行宽 100。
- 随机性：所有随机源从 `seed.py` 取（torch / numpy / python random 三处同时设），实验脚本开头 `seed_everything(seed)`。
- 关键模块必须有单测：`benchmark.py`（对拍）、`losses.py`（CVaR 手算）、`backtest.py`（单窗口手算）、`stats.py`（bootstrap）。
- 不许 `TODO` 留在合并的代码里；未完成的写成 `raise NotImplementedError` + issue 编号。

---

## 9. 实验登记（防止报告写不出来）

`docs/experiments.md` 一张表，**每跑完一个实验追加一行**，不许攒到最后补：

| exp | seed/eps/cost | commit | 运行时长 | 关键结果（一句话） | 是否达标 |
|---|---|---|---|---|---|
| e1 | seeds 0–4, ε=0 | c3bc2ab | 32 min | price 1.9431, delta RMSE 0.105 | ✅ |
| e1d1 | seeds 0–4, ε=0, d=1 | c6ef7da | 26 min | price 2.5338, simpler 2.5224 | ✅ |
| e3 | seeds 0–4, ε=0.01 | a1b2c3d | 3.1 h | recurrent 5.51 vs simpler 6.74 | ✅ |

- "是否达标"只写 ✅ / ❌ / ⚠️（⚠️=数值在区间内但机制可疑，要写备注）。
- 报告的每一张图、每一行数字都要能在这张表里找到来源行。

---

## 10. 图表与报告规范

> 图先做齐（8 张全出，进 `figures/`），报告篇幅与取舍等写报告时再定（§13）。

- 图：300 dpi、PNG、英文轴标签与图例、无标题（标题进 caption）、网络 vs 基准用固定配色（网络=蓝、基准=灰、论文参考=红虚线）。
- 图必须自解释：caption 一句话结论（"有成本时 recurrent 明显优于 simpler"），不是"Figure 5"。
- 表：3 位有效数字；多 seed 报 `均值 ± 标准差`；与论文数值并列时**必须加一列"论文值"**。
- 正文：英文；每节先一句话结论，再证据；所有"deep hedging 更好/更差"必须带 C\* 数值和 CI。
- 引用论文数字时标注小节（如 "paper §5.2, Fig 5"）。

---

## 11. 检查点（Gate，按依赖顺序，不排日期）

| Gate | 条件 | 谁负责 | 不过怎么办 |
|---|---|---|---|
| G0 | 骨架可跑、CI 绿 | A1 | 阻塞所有人，优先修 |
| G1 | B0 三个对拍通过 | A1 | 走 plan §2.5 降级路径 3，只砍 E1 对比 |
| G2 | E1 出图 + `policy_heston_d1` 交付 | A1 | B2-M1 无法开始，B2 先做 Bench 腿 |
| G3 | E2/E3 出表 | A2 | 报告缩小到 E1+E4 |
| G4 | E4 两条斜率 + 低摩擦附表 | A2 | 只交 BS 那条，写明 d>1 未复现 |
| G5 | 港股数据 + 校准 + 生成器 | B1 | B2 用历史 bootstrap 情景先跑 |
| G6 | 港股主表齐（C\*95/C\*99） | B2 | 结论降级为方向性 |
| G7 | 报告定稿 | 全员 | — |

**G1 不过不许开 E1 训练**；**G5 不过不许跑 M2**。

---

## 12. 变更记录

| 版本 | 日期 | 改动 | 影响 |
|---|---|---|---|
| v1.0 | — | 初版：目录、命名、接口、DoD、Gate | — |
| v1.1 | — | 三项拍板：`e4low` 先 seed 0–2、M2 主口径参数化 GARCH、图先做齐（§13） | 不阻塞开工 |
| v1.2 | 2026-11-14 | 冻结 d=2 输入特征 f=3 与网络宽度（§5.1）；成本按名义本金并写明 d=2 两腿量级差（§5.1）；`env_bs.py` 归属 A1（§2、§6） | E4-BS 复用 A1 环境；E3 结论口径统一 |

---

## 13. 已定的三项（v1.1 拍板）

| 项 | 决定 | 备查 |
|---|---|---|
| `e4low` seed 数 | **先跑 seed 0–2**；若 `(p_ε−p0)` 与 seed 间标准差区分不开再补 3–4 | 省一半算力，需要时才补 |
| M2 路径生成主口径 | **参数化 GARCH 情景**；历史 bootstrap 情景只作 robustness 附表 | 与论文「校准后模拟」同路 |
| 报告篇幅 / appendix | **暂不定；图先做齐（8 张全出）**，写报告时再取舍 | `figures/` 的图全部保留 |

余下只剩报告篇幅与 appendix 形式，写报告时定，不阻塞任何任务。
