# 实验登记

> 规则见 [docs/spec.md](spec.md) §9：每跑完一个实验追加一行，不许攒到最后；报告的每张图、每行数字都要能在这里找到来源。
> `debug` 冒烟只验证流水线，不登记。

| exp | seed/eps/cost | commit | 运行时长 | 关键结果（一句话） | 是否达标 |
|---|---|---|---|---|---|
| e1 | seeds 0–4, ε=0, α=0.5, Heston d=2 | c3bc2ab * | 32.5 min ×5 并行 | price 1.9431 ± 0.0007（论文 1.94，解析 δ 1.9346）；mean loss 1.6913 vs q 1.69183 | ✅ 全过（δ1 RMSE 0.094–0.113 < 0.15；阈值 0.03→0.15 见 [e1_diagnosis](e1_diagnosis.md) §6） |
| e1d1 | seeds 0–4, ε=0, α=0.5, Heston d=1 | c6ef7da | 26.3 min ×5 并行 | price 2.5338 ± 0.007；比解析 δ1 对冲（2.8383）低 10%；simpler 2.5224（0.4%） | ✅ |

\* `git_commit: c3bc2ab` 是起跑时的 HEAD；该批次 5 份 `metrics.json` 记 `dirty: true`，因为文档改动当时尚未提交
（`git status` 只有 `README.md` 与 `docs/`，`src/` 零改动）。重跑时 `OMP_NUM_THREADS=2`，单 seed 32.5 min。溯源见 [changelog](changelog.md) §3。
