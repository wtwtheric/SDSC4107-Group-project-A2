# 变更记录与待办（2026-11-14）

本文件记录：① 2026-11-14 这一批文档修订（对应 plan v1.4）；② E1 重跑（2026-11-14 13:52）的结果、同步记录与 dirty 溯源。
（代码零改动：ε、batch、阈值都不在代码里，见 §1.3。）

---

## 1. 这批改了什么

### 1.1 触发原因

队友指出 `docs/plan.md` §1.2 的 ε 应为 `2^{-(i+5)}`。对照原文（`docs/paper/deephedging.pdf` = arXiv 1802.03042v1）
与 LaTeX 源码（`refs/deephedging_paper_source.tex`）复核：**队友对，plan 错**。顺带把全区其它口径一并核了一遍。

### 1.2 改动清单（全部是文档）

| 文件 | 位置 | 改动 | 依据 |
|---|---|---|---|
| `docs/plan.md` | L1/L3 | 标题 v1.4；状态「待开工」→「进行中（B0 ✅、E1 ✅、E1-d1 ✅）」 | `results/b0`、`results/e1`、`results/e1d1` |
| `docs/plan.md` | L18 | 第 (3) 条：ε 的「极端区间」→ **小成本区间 2^{-(i+5)}（156→9.8bp）** | tex L1781；Fig 10 |
| `docs/plan.md` | L20 | 新增 **v1.4 修订记录**（六条） | — |
| `docs/plan.md` | §1.1 表 5.3 行 | 工具补「d=1 标的 / d=2 标的+方差互换」；ε 网格 → `{2^-6,…,2^-10}` | tex §5.3 |
| `docs/plan.md` | §1.2 第 2 条 | 整段重写：ε_i=2^{-(i+5)} = 156.3/78.1/39.1/19.5/9.8bp，验证 O(ε^{2/3}) 小 ε 极限 | tex L1781；Fig 10 坐标反解 |
| `docs/plan.md` | §2 约定 | 方差互换价格 S²_0 = **bT ≈ 0.0033**（不是 0.04）；两腿单位成本比 ≈3×10^4 | 论文式 (5.4) |
| `docs/plan.md` | E1 判据 | delta 面 RMSE **< 0.15**（原 0.03 不可达） | `docs/e1_diagnosis.md` §6 |
| `docs/plan.md` | E4 设定/风险 | ε 网格 `{2^-6..2^-10}`，bp 值；风险方向反转（小 ε ⇒ p_ε−p0 更小 ⇒ 要更多路径与 seed，非更大 batch） | 式 (5.8) |
| `docs/plan.md` | 低摩擦附表 | `{0.1,0.03,0.01}` → **向下扩展** `{2^-11,2^-12,2^-13}` ≈ 4.9/2.4/1.2bp | 论文低端已到 9.8bp |
| `docs/plan.md` | §2.5 / §4 / §4.5 | 方差互换闭式口径；ETF 腿 **~30bp（25–40bp）**；2800.HK **无印花税**（ETF 自 2015-02-13 豁免） | IRD FAQ；pfhedge 0.23.0 |
| `docs/plan.md` | §10.3 / §10.4 | 主表照抄 `2^{-(i+5)}`；varswap_k 取论文 (5.4) 的 **S²_k**（S²_0=bT≈0.0033）；删「不往下探到 1e-3」 | 论文式 (5.4)；`docs/method_replication.md` §2 |
| `docs/plan.md` | §6 | 补「v1.4 实际仓库」文件清单 | 仓库实况 |
| `docs/spec.md` | L170/L173 | varswap_k 口径同 plan；成本比 3×10^4 | 同上 |
| `docs/spec.md` | §9 示例行 L325-326 | ε=16 / delta RMSE 0.021 → ε=0 / RMSE 0.118（重跑后已再改为 0.105，见 §4）；e4bs→e3 | 实际结果 |
| `docs/spec.md` | L186 示例 JSON | `git_commit: "a1b2c3d"` 占位符（已换成真实 commit `c3bc2ab`，见 §4） | §4 待办 7 |
| `README.md` | L24 | 方案版本 v1.3 → v1.4 | — |
| `docs/paper_fulltext_v1.txt` | 新增文件 | 机器抽取全文（91KB），顶部加了 4 行「LOSSY、负号会丢、精确请看 tex」的警告 | — |

### 1.3 为什么代码不用改

- ε 是标量配置：`src/common/config.py` 的 `eps` → `src/common/metrics.py:47 proportional_cost(...)` →
  `env_heston.py:40` / `env_bs.py:25` 透传；全仓库无 ε 网格硬编码。E3 用 ε=0.01（论文原值），E4 属 A2。
- 方差互换口径本来就按论文实现在 `env_heston.py:92-107`（`∫V` 条件期望 + `L(t,V)`），plan/spec 的文字此前与代码不一致，这次是把文字改成和代码一致。
- `e1.py:221` 的 checks 阈值早已是 0.15；`.git` 里 `results/e1/*/metrics.json` 是旧跑（0.03）留下的，属元数据。

---

## 2. 核查结论（可直接写进报告的表述）

1. **ε 网格**：论文 §5.3 用 `ε_i = 2^{-(i+5)}`，i=1..5 → 156.3 / 78.1 / 39.1 / 19.5 / 9.8 bp，是**小成本极限**，验证式 (5.8) `p_ε − p_0 = O(ε^{2/3})`。
   证据：tex L1781 `$\varepsilon_i=2^{-{i+5}}$`；Fig 10 五个点的横坐标从绘图指令反解为 ln(2^-10…2^-6) = −6.931…−4.159（误差 <0.002），
   横轴标题为 `log(ε)`，**自然对数**；换底只改截距、不改斜率 0.67。PDF 上标少了括号（印成 −i+5）才导致误读。
2. **斜率**：BS 0.67、Heston 0.71（`asymptoticsBSFS.pdf` / `asymptotics.pdf` 图内注释）。
3. **方差互换价格**：S²_t = E_t[∫_0^T V_s ds] = 已实现段 + L(t,V)，S²_0 = bT = 0.0032877（B0 实测 0.0032992，0.350%）。
   0.04 是年化 fair strike（S²/T），若当输入会引入漂移、网络套利，价格塌到 1.34（见 `docs/method_replication.md` §2）。
4. **港股 ETF 成本**：2800.HK 等 ETF 自 2015-02-13 起免印花税（0.13% 只针对正股）；往返 ~30bp 来自佣金 6–10bp + 征费 ≈1.7bp + 1 tick ≈20bp。
5. **delta RMSE**：纯监督拟合解析 δ 的极限只有 ~0.065（`docs/e1_diagnosis.md` §6），0.15 只作「跑飞」护栏（BN 失稳曾 8.04）。
6. **图号映射**：渲染 Fig 2/3/4 = tex `fig0` / `15daysDeltaS` / `PaperFig1`；Fig 5/6/7 = `PaperFig2/3/4`；Fig 10/11 = `asymptoticsBSFS` / `asymptotics`。plan 原描述无误。

---

## 3. E1 重跑（已完成，2026-11-14 13:52）

**动机**：`results/e1/*/metrics.json` 记录 `dirty: true`（spec.md:153 规定脏工作区的数字不许进报告），
且 checks 仍是旧阈值（`delta_rmse_lt_0.03: false` → `passed: false`），与代码/文档里的「✅」矛盾。

**实际执行**：5 进程 13:19:30 起跑（`$env:OMP_NUM_THREADS="2"`，`src.replication.train --config configs/exp/e1.yaml --seed 0..4`），
13:52:01–13:52:04 全部落盘，runtime 1951–1953 s（32.5 min）。13:53:30 的关机事件在最后一次写盘之后，无半截文件。
**文档提交那一步没有执行**：`git reflog` 显示 HEAD 仍是 11:08:51 的 c3bc2ab；起跑时 `git status` 已有
`M README.md`、`M docs/plan.md`、`M docs/spec.md`、`?? docs/paper_fulltext_v1.txt`（changelog 跑到一半才建）。
因此 5 份新 metrics 记 `git_commit: c3bc2ab`、`dirty: true`；porcelain 里没有任何 `src/` 路径，代码本身与 c3bc2ab 一致。
按决定不重跑，`dirty` 如实保留（把 JSON 改成 false 等于伪造出处）。

```powershell
# 下次要干净出处时（32.5 min ×5 并行）
git add -A && git commit -m "docs: plan v1.4 + E1 重跑结果"
$env:OMP_NUM_THREADS="2"
0..4 | ForEach-Object { Start-Process .\.venv\Scripts\python.exe `
  -ArgumentList "-m","src.replication.train","--config","configs/exp/e1.yaml","--seed","$_" -NoNewWindow }
```

**结果**（`results/e1/{0..4}/metrics.json`）：

| seed | price | mean loss | CVaR99 | simpler | δ1 RMSE |
|---|---|---|---|---|---|
| 0 | 1.9438 | 1.6927 | 3.548 | 1.9437 | 0.1120 |
| 1 | 1.9435 | 1.6914 | 3.518 | 1.9431 | 0.0993 |
| 2 | 1.9427 | 1.6908 | 3.580 | 1.9461 | 0.1129 |
| 3 | 1.9421 | 1.6901 | 3.521 | 1.9435 | 0.1073 |
| 4 | 1.9431 | 1.6913 | 3.563 | 1.9443 | 0.0943 |
| **mean** | **1.9431** | **1.6913** | 3.546 | 1.9441 | 0.105 |

price 样本 sd 6.6×10^-4；δ1 RMSE 0.0943–0.1129；四个数字判据加 `delta_rmse_lt_0.15` 全 true，`passed: true`。
与旧跑（1.9432 ± 0.0006，60.4 min，`18a2734(dirty)`）比：均值差 1×10^-4，逐 seed 漂移 ≤5×10^-4——
训练不是逐位确定的（BLAS 归约顺序），差值在 seed 噪声以内，不构成配方变更的证据。

---

## 4. 重跑后同步的文件（已全部完成）

| # | 文件:位置 | 已改成 | 状态 |
|---|---|---|---|
| 1 | `results/e1/*/metrics.json` | `git_commit: c3bc2ab`、`dirty: true`（仅文档未提交）、`delta_rmse_lt_0.15: true`、`passed: true` | ✅ 自动生成 |
| 2 | `docs/experiments.md` L8 | `c3bc2ab *` / `32.5 min ×5 并行` / 「✅ 全过（δ1 RMSE 0.094–0.113）」+ `*` 脚注说明 dirty 来源 | ✅ |
| 3 | `docs/spec.md` L186、L325 | `a1b2c3d` → `c3bc2ab`；§9 行改 `1.9431 / RMSE 0.105 / 32 min` | ✅ |
| 4 | `docs/method_replication.md` L208 | 「clean working tree」→ 记 commit + dirty 标志，并注明 E1 批次仅文档未提交 | ✅ |
| 5 | `docs/method_replication.md` §9.1 表 + L226-235 | 逐 seed 换新数值（mean 1.9431 / 1.6913 / 0.105）；0.45%→0.44%、0.15%→0.06%、1.9432→1.9431 | ✅ |
| 6 | `docs/e1_diagnosis.md` §5 seed 表 | 同上；`passed=False` 的说明改成「旧阈值遗留，重跑后已 true」 | ✅ |
| 7 | `README.md` L67、L72-73 | 1.9431 ± 0.0007 / mean loss 1.6913；「单核 60 min」→「2 线程 32 min」 | ✅ |
| 8 | `docs/plan.md` L3 | 「E1 ✅（c3bc2ab）」 | ✅ |
| 9 | 本文件 §3 | 改写成「已完成」，含新结果表与 dirty 溯源 | ✅ |

**结论层面（三句话，已按实际改写）**：

1. 「E1 ✅ **除 δ1 RMSE 判据外**全过」→ 「E1 ✅ **全过**」（δ1 RMSE 0.094–0.113 < 0.15，阈值理由见 `e1_diagnosis` §6）。
2. **e1d1 已重跑干净**（`c6ef7da`，26.3 min，2.5338 ± 0.007）：见 §5；`dirty: false`，可进报告。
3. spec.md §5.2：E1 的 checks 现在全 true，数字可引用；唯一保留的瑕疵是 `dirty: true`（起跑时仅文档未提交，
   `src/` 零改动），已在 `experiments.md` 与本节注明——若审阅方严格要求 `dirty: false`，唯一办法是重跑 32.5 min。

---

## 5. E1d1 干净重跑（2026-11-14 14:30，已完成）

**做法**：文档与图提交后工作区干净，14:04:12 起 5 进程（`OMP_NUM_THREADS=2`，
`--config configs/exp/e1d1.yaml --seed 0..4`），runtime 1574–1579 s（26.3 min），
5 份 metrics 全部 `dirty: false`、`git_commit: c6ef7da`、`passed: true`。

| seed | price | mean loss | CVaR99 | simpler | analytic δ1 hedge |
|---|---|---|---|---|---|
| 0 | 2.5291 | 1.6923 | 8.567 | 2.5163 | 2.8363 |
| 1 | 2.5290 | 1.6923 | 8.590 | 2.5186 | 2.8454 |
| 2 | 2.5276 | 1.6907 | 8.418 | 2.5295 | 2.8355 |
| 3 | 2.5429 | 1.6999 | 8.586 | 2.5214 | 2.8399 |
| 4 | 2.5402 | 1.6990 | 8.573 | 2.5261 | 2.8344 |
| **mean** | **2.5338** | **1.6948** | 8.55 | 2.5224 | 2.8383 |

与旧跑（2.5337 ± 0.007，`fe04fe0(dirty)`，31.8 min）比：均值差 1×10^-4，逐 seed ≤8×10^-4，在 seed 噪声内；
结论（比解析 δ1 对冲低 ~10%、simpler 差 0.4%、CVaR99 8.5）全部不变。

顺手补了一次重算：§9.2 的「δ1 RMSE 0.24（MAE 0.19）」原先没有脚本可复现（`pnl.npz` 只存 loss）。
新增 `scripts/recompute_d1_delta.py`（seed 0、2×10⁴ 条新路径、k=15、S∈[90,110]）：新权重得
**RMSE 0.2432 / MAE 0.2001**（旧权重 0.2408 / 0.1958），5 个 seed 的 RMSE 为 0.198–0.307。
文档已改为「RMSE 0.24（MAE 0.20）」并补上协议与 seed 范围。

`scripts.export_policy_d1` 已用新 seed 0 重新导出 `src/policy/weights/heston_d1.pt`：meta 里
`source_git_commit` / `export_git_commit` = c6ef7da，`eval_metrics` 与新结果一致；`tests/test_policy_d1.py` 2 passed。
已同步 `experiments.md`（e1d1 行）、`method_replication.md`（§8 出处、§9.2 表与两处百分比）、`plan.md` L3、`spec.md` §9。

---

## 6. 遗留 / 未做

- **E1 的 `metrics.json` 是 `dirty: true`**：起跑时仅文档未提交（`src/` 零改动），见 §3；要 `dirty: false` 需重跑 32.5 min。
- **`python -m src.repro` 不存在**：plan §8 的「一键出表」验收项还差 `src/repro.py` + `reports/`。
- **batch 10⁴ 校准实验**：未跑。若跑：固定 3×10⁷ 总路径，`batch 4096 × 7500 iter`（cosine 重设）对照 `1024 × 30000`，各 2 seed。
- **GPU**：未装 cu128。结论是 E1 不需要（overhead-bound：float32 都不比 float64 快，见实测）；A2/E4 再考虑「大 batch + float32 + 场景池 + jit」。
- **线程写入 config**：`train.py` 目前只能靠 `OMP_NUM_THREADS` 环境变量，未进 git。可在 A2 前加 `threads:` 配置项。

---

## 7. A1 报告 + 仓库整理（2026-11-14 15:00）

**新增 `docs/report_a1.md`**（A1 报告）：六项交付对照 → 方法（市场/定价与基准/网络/训练/验收检查）→
结果（E1、E1d1、Fig 2/3/4）→ 复现命令 → 与论文的 deviation 表 → 已知限制 → 附录（产物出处、时间线、体积）。
所有数字取自 `results/e1/{0..4}/metrics.json`、`results/e1d1/{0..4}/metrics.json` 与 `results/b0/b0.json`，
可直接重算；README 已加链接。

**顺手修掉的陈旧表述**：
- `docs/e1_diagnosis.md` 头部状态「未解决」→「已解决（§6 定稿）」，与 5 seed 结果一致（正文其余部分保留诊断过程）。
- `README.md` 目录树：补 `changelog.md` / `report_a1.md`，`reports/` 标注「A2 起建」，scripts 行补 `recompute_d1_delta`，
  环境体积 828 MB → 约 1.0 GB。

**仓库清理**（删的都是 gitignore 里可再生成的东西，共 29 MB + 缓存）：
- `results/e1d1_prev/`（12.8 MB）：E1d1 干净重跑前的备份副本，新结果已确认并入库文档。
- `results/{e1,e1d1}/_seed*.log`（20 个）：起跑重定向的临时 stdout/stderr（err 全 0 字节，内容已被 `run.log` 覆盖）。
- `results/b0/heston_surface.npz`（5.2 MB）：旧版定价面缓存，`benchmark.SURFACE_CACHE` 只读 `heston_surface_v2.npz`。
- `.pytest_cache/`、`.ruff_cache/`、`__pycache__/`：工具缓存。

**保留判断**（不是垃圾，别顺手删）：
- `docs/paper_fulltext_v1.txt`（91 KB）：机器抽取的论文全文，顶部有 LOSSY 警告，是口径修正的引用依据。
- `refs/`（120 MB）：pfhedge / deephedging 只读快照，README 已说明「不改、不提交」。
- `results/e1/*/pnl.npz`（约 117 MB）：Fig 2/3 与 δ 重算的原料；删了就得重跑 32 min×5。
- `results/sweep/`（1.7 MB）：8 个配方对照的原始记录（`e1_diagnosis.md` §6 的表）。

---

## 8. 上线 GitHub + 修远端 CI（2026-11-14 16:30）

**仓库**：<https://github.com/Casalba/SDSC4107-Group-project>（public，默认分支 `master`）。
首次 push 为 `cd975de`（187 objects / 3.41 MiB）；`.venv/`、`results/`、`refs/` 均在 `.gitignore` 内未上传，
远端 tree 与本地 `git ls-files` 逐文件一致（71 个）。

**远端 CI 修复**：首跑（`cd975de`）红在 CI 步骤——`ci.yml` 的 Install 只装 torch / pytest / ruff / numpy / scipy / pyyaml，
漏了 `pfhedge`，而 `src/replication/env_heston.py` 顶层 `from pfhedge.stochastic import generate_heston`，
pytest 收集期即 ImportError（exit 2）。`d0024f3` 在 Install 步骤补 `numpy==1.26.4`、`scipy==1.13.1`、`pfhedge==0.23.0`
（pfhedge 要求 numpy<2，而 numpy>=2 才有 scipy>=1.18，二者互斥；版本与 `requirements.lock.txt` 一致）。
重跑 CI 绿（Install + CI 两步 success）。README 顶部加了 CI badge。

**凭据**：Git Credential Manager 已存 GitHub 凭据，之后 agent 可直接 push。


---

# E2 output/evaluation workflow update (2026-11-14)

- Added pathwise positions and explicit hedging-error arrays to the E2 100k output. Positions are archived as float32 to limit file size; terminal losses remain float64.
- Added a chunked frozen-policy evaluator for a fresh 1,000,000-path common test sample. It records tail diagnostics and per-step position summaries without retaining all pathwise positions.
- Added canonical Figure 6/7 rendering to repository-root `figures/`, plus a CSV summary in the evaluation directory.
- Clarified sign conventions: loss is `L = Z - hedge_gain`; Figure 7 plotted hedging error is `q - L`; CVaR is calculated on the upper tail of `L - q`.
- Added opt-in `backups/` for selected checkpoints, metadata, and datasets. Working `results/` and `data/` remain ignored. Large binary backups use Git LFS; anything committed to this public repository is public.
- Added Kaggle clone, environment, evaluation, figure-generation, and push instructions in `docs/e2_kaggle.md`.

Validation status: code changes are prepared on `exp/e2-output-eval`. Runtime pytest and Kaggle GPU evaluation must still be run before treating the outputs as validated results.


---

## E2cs five-seed robustness completion (2026-11-19)

- Archived corrected simpler-policy E2cs seeds 1–4 under `backups/e2cs_seeds1_4/`; seed 0 remains under `backups/e2cs_simple_seed0/`.
- Consolidated the original five-seed checkpoint results in `docs/e2cs_multiseed_summary.md` and the machine-readable aggregate in `reports/e2cs/e2cs_multiseed_summary.json`.
- The alpha=.99 policy reduces CVaR99 relative to the alpha=.95 policy in 5/5 seeds by 0.132714 ± 0.033466 on the primary one-million-path evaluations.
- Seed 3 is the main alpha=.99 numerical outlier, but retains the same direction of tail-risk improvement.
- Independent post-training BN recalibration across all five seeds strongly reduces tail-risk dispersion while preserving the flatter, left-shifted alpha=.99 hedge. Original checkpoints remain the formal replication outputs; recalibration remains a diagnostic.
- E2cs is frozen. No further retraining is required solely to match Figures 8–9.


---

## E3 five-seed robustness completion (2026-11-20)

- Archived and reviewed the frozen QE-M/current-input E3 baseline for seeds 0–4 under `backups/e3/`.
- Added `docs/e3_multiseed_summary.md` and `reports/e3/e3_multiseed_summary.json`.
- The recurrent policy has a lower independently calibrated price in 5/5 seeds. The five-seed price gap is 0.057864 ± 0.003902.
- The recurrent raw mean-loss advantage is almost entirely explained by lower transaction costs; the pre-cost mean-loss difference is only 0.000225 ± 0.000131.
- Centered Figure 5 distributions overlap strongly because each policy is plotted at its own independently calibrated price. The overlap is stable across seeds, not an isolated failed run.
- Completed the three simulator/input fidelity controls. Paper inputs raise both E3 prices toward the paper; the simulator-only change has little effect. None recovers the paper's large 1.2309 recurrent–simpler gap.
- E3 is frozen as a stable qualitative but incomplete quantitative reproduction. No further retraining is required solely to force agreement with Figure 5.

---

## A2 E2–E4 formal freeze and integration closeout (2026-11-21)

- Added the separate A2 registry `docs/experiments-e2-e4.md`; the A1-owned `docs/experiments.md` remains unchanged.
- Marked E2 core, E2 call spread, E3, E4-BS and E4-Heston seeds 0–4 as complete and frozen. “Frozen” records completion without treating paper-target discrepancies as passes.
- Updated `docs/plan_A2_next.md` and `docs/method_replication-A2.md` to remove obsolete statements that E2 was still running or that E4 seeds 1–4 remained pending.
- Recorded the deliberate exclusion of large pathwise `pnl.npz` files to limit repository size and clone time.
- Audited the repository branch graph. The reviewed A2 branches are ancestors of `exp/e2e3-fidelity-controls`; the older divergent `feat/e3-independent-eval-fig5` implementation is superseded and must not be merged separately.
- Retargeted PR #12 from `exp/a2-seed0-batch` to `main` and updated its title/body to describe the complete E2–E4 scope.
- Local `python scripts/ci.py` passed after fetch/pull. Final GitHub Actions CI on the documentation head remains the merge gate.
