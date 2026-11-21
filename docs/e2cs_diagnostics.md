# E2cs corrected simpler-policy seed-0 diagnostics

> **Multi-seed update (2026-11-19):** corrected E2cs seeds 0–4 are complete. See [E2cs five-seed robustness summary](e2cs_multiseed_summary.md). The sections below preserve the detailed seed-0 diagnostic history.

## Scope and provenance

This analysis exclusively uses the corrected E2cs simpler-network seed-0 checkpoints in `backups/e2cs_simple_seed0/e2cs/0.95/0` and `.../0.99/0`. The original batch E2cs recurrent attempt is excluded. The correction matters because Figures 8–9 of *Deep Hedging* compare the model hedge against a per-time-step, feed-forward policy based on the current spot price and instantaneous variance, `F_k(S_k,V_k)`.

The accepted run uses `recurrent: false`, BatchNorm, 17 hidden units, stock and variance-swap hedge legs, normalized call-spread strikes 100/101, zero proportional trading costs, and k=15 for the surfaces. We preserve the trained weights and do not rerun training or the 1M-path OOS evaluation.

The archived `metrics.json` reports the following independent-of-rendering results:

| Measurement | 95%-CVaR policy | 99%-CVaR policy |
|---|---:|---:|
| Own-alpha price | 0.958340 | 0.963278 |
| Mean raw terminal loss | 0.609570 | 0.609472 |
| Common OOS CVaR 95% | 0.958340 | 0.922781 |
| Common OOS CVaR 99% | 1.087331 | 0.963278 |

The 99%-trained policy has an 11.41% lower realized CVaR99 on common OOS paths than the 95%-trained policy. That supports tail-risk improvement but does not prove the *shape* of the NN hedge replicates the paper.

## Figure style and comparability

`scripts/a2/make_e2cs_figures.py` uses blue-white-red (`bwr`) rather than the grey-centered `coolwarm`. The same model/NN hedge color interval is used across Figures 8 and 9. Differences use another shared symmetric interval centered at **zero**, which remains white; blue indicates NN below the model and red indicates NN above it.

The model/NN hedge color scale is **not** forced to center on zero, avoiding a misleading grey/white block for small but meaningful stock positions. The 3D renderer uses `shade=False` to avoid lighting-driven color changes. Because the hedges are not signed differences, their color should not be interpreted as two sides of a risk sign.

The original figure assets remain untouched; diagnostics are written to `figures/e2cs_diagnostics/` until reviewed.

## Hedge-shape diagnostics

`scripts/a2/diagnose_e2cs.py` reloads the frozen policy checkpoints with BatchNorm in evaluation mode and directly evaluates the same 81-by-61 uniform `(S,V)` grid used by the original Figures 8–9. It computes the analytical-model hedge using the existing Heston finite-difference pricer. It records **both** the stock and the variance-swap legs.

For each alpha and hedge leg it reports:

- Grid RMSE/MAE, mean signed deviation, correlation and maximum absolute deviation
- Model peak and NN peak locations and heights
- NN hedge at the model's own peak and the difference at that location
- Range of NN and model hedge across stock prices at low and median variance
- Percentage of grid cells where the NN is below the model and where the NN is negative

These are **uniform-grid statistics**, not probability-weighted OOS error measures. They describe policy-shape reproduction over the deliberately chosen state grid. Large errors in remote or sparsely visited states may overstate errors on actual paths. A further path-support-weighted diagnostic should be added if the original large `pnl.npz` arrays become available.

These diagnostics help distinguish a genuine global flattening of the 99%-policy stock hedge from a shifted peak, a relative scale change, or a limited failure near `S≈100,V≈0.02`. A lower risk score can coexist with a less faithful model-delta shape because CVaR optimization is not an objective to match model delta pointwise.

## Reproduce without retraining

From repo root:

```bash
python -m scripts.a2.diagnose_e2cs --device cpu
```

Or via the unified A2 entry point:

```bash
python -m scripts.a2.make_figures_a2 --only e2cs-diagnostics --seed 0
```

Outputs:

- `reports/e2cs/e2cs_shape_diagnostics.json` and `.csv`
- `reports/e2cs/e2cs_diagnostic_grids.npz` (local-only, excluded by Git)
- `figures/e2cs_diagnostics/fig08_e2cs_stock_position.png`
- `figures/e2cs_diagnostics/fig09_e2cs_stock_position.png`
- Matching blue-white-red heatmaps

These scripts do not simulate paths, train policies, change source metrics or overwrite original backups. Both checkpoints must be genuine weights rather than unmaterialized Git LFS pointers. If running inside AutoDL, `git lfs pull` is required when the checkpoints were checked out as pointer files.

## Measured seed-0 checkpoint diagnosis (2026-11-17)

The new diagnostic script has now been executed in GitHub Actions using the actual corrected simpler seed-0 checkpoint files (not mocked networks or a regenerated training run). The 81 × 61 uniform state grid produces:

| Leg and metric | Alpha 0.95 | Alpha 0.99 |
|---|---:|---:|
| Stock hedge RMSE against model | 0.0307 | 0.0608 |
| Stock hedge grid correlation | 0.7808 | 0.0167 |
| NN stock hedge maximum | 0.3242 | 0.1166 |
| NN stock hedge spot at peak | 99.4 | 96.8 |
| NN stock hedge at model peak (S=99.8, V=0.02) | 0.2987 | 0.0511 |
| Stock hedge deviation at model peak | -0.0728 | -0.3205 |
| Variance swap hedge RMSE against model | 29.48 | 58.26 |
| Variance swap hedge correlation | 0.8991 | 0.4260 |
| NN variance swap maximum | 335.05 | 93.00 |
| NN variance swap at model peak (S=99.3, V=0.02) | 267.17 | 1.88 |

The shared model stock hedge maximum is 0.3715 at S=99.8, V=0.02. The model variance-swap maximum is 332.29 at S=99.3, V=0.02.

**Inference.** The alpha=0.95 simpler policy largely reproduces the model hedge's stock peak and variance-swap structure, albeit with measurable deviations. In contrast, the alpha=0.99 policy shows a large downward scale reduction **and** a leftward shift of the stock peak. The stock grid correlation drops to approximately zero; the variance-swap leg also loses much of the model peak structure. This is a genuine checkpoint-derived shape deviation, not an artifact introduced by the old grey-centered colormap. The leftward peak shift is *qualitatively compatible* with more conservative effective-strike/barrier behavior, but we have not identified a precise effective barrier or established the paper's quantitative hedge-surface reproduction.

These diagnostics do not contradict the 11.41% lower realized CVaR99 under common OOS paths: minimizing tail losses is a different objective from matching the risk-neutral model hedge pointwise. The grid is not weighted by the realized state distribution. We must check the actual k=15 state support (and compare training-path conditioning / BatchNorm statistics) before deciding whether retraining is warranted.

The four checkpoint-derived rows are archived in `reports/e2cs/e2cs_shape_diagnostics_seed0.csv`. The full JSON, grid NPZ, and updated figures are available in the successful GitHub Actions artifact:

https://github.com/wtwtheric/SDSC4107-Group-project-A2/actions/runs/44038798249/artifacts/14307724287

**Verification:** GitHub Actions CI completed successfully with 65 passing tests. The actual checkpoint-only shape diagnosis completed successfully and produced the plots and data artifact; no model training was performed.

## Path-support, BatchNorm and training-convergence investigation (2026-11-18)

A checkpoint-only and independent-path diagnostic has been implemented as `scripts/a2/diagnose_e2cs_support_bn.py`, using the original simple-policy seed-0 weights; the same Heston model parameters and k=15. The initial successful run uses **40,000 newly generated CPU Heston paths, simulation seed 928371**, not the original GPU visual paths in the omitted `pnl.npz`. Results from this run are in GitHub Actions [run 44259116837](https://github.com/wtwtheric/SDSC4107-Group-project-A2/actions/runs/44259116837); the later regional-error extension is separately verified by CI before acceptance.

### Market support

On the independent 40k sample the model/NN surface grid covers spot `[95, 103]` and variance `[0.02, 0.14]`. Marginal coverage is **76.46% in spot** and **22.62% in variance**, but **only 16.15% jointly**. The variance median is **exactly zero**; the simulator produces a large boundary-zero component under this high-vol-of-vol Heston specification. The model-stock maximum neighborhood `S∈[99.3,100.3], V∈[0.02,0.04]` is occupied by **0.74%** of all paths; the 99%-policy shifted peak neighborhood `S∈[96.3,97.3], V∈[0.02,0.04]` is occupied by **0.42%**. Hence the uniform grid heavily emphasizes a portion of the state space visited relatively rarely. It must not be used as a population-weighted estimate of the policy's hedge error.

### Conditional path-weighted hedge comparisons

Within the 16.15% of simulated states inside the plotting grid, the model delta is bilinearly interpolated from the existing Heston finite-difference grid; the NN policy is evaluated directly:

| Within-grid metric | α=0.95 | α=0.99 |
|---|---:|---:|
| Stock RMSE | 0.03721 | 0.07818 |
| Variance-swap RMSE | 33.90 | 74.51 |
| Stock mean signed NN-model difference | -0.00518 | -0.04968 |
| Variance-swap mean signed difference | +5.47 | -24.21 |
| Fraction of within-grid states with NN stock below model | 69.17% | 88.03% |

The 99%-CVaR policy's poorer hedge-surface fidelity persists under the *within-grid path distribution*. **However, this comparison ignores 83.85% of paths outside that grid**, which requires a different model-delta calculation or a wider carefully validated state grid before claiming global OOS shape errors.

### BatchNorm checkpoint evidence

For the k=15 step, both policies' BN1 and BN2 track **30,000 batches**, consistent with 30,000 optimizer iterations. Comparing actual independent-state pre-BN activations against frozen running moments:

| Statistic (median across neurons) | α=0.95 | α=0.99 |
|---|---:|---:|
| BN1 standardized absolute mean shift | 0.0919 | 0.1377 |
| BN2 standardized absolute mean shift | 0.0339 | 0.5391 |
| BN1 live/running variance ratio | 1.0275 | 1.0300 |
| BN2 live/running variance ratio | 1.0202 | 0.9342 |

The 99% BN2 mean shift is notably larger. Replacing the frozen running moments by moments from the independent sampled batch in a *non-mutating counterfactual* changes α=0.95 stock positions by mean absolute **0.01885** and α=0.99 by **0.02984**; variance-swap mean absolute changes are **14.62** and **37.95**. This flags greater BN sensitivity under α=0.99, but does **not** establish BN as the cause of weaker hedge-shape replication or imply that using batch statistics in production would improve CVaR. The latter would require a controlled out-of-sample evaluation.

### Training history (not convergence proof)

Window-mean stochastic training CVaR declines over 30k iterations for both policies, α=0.95 **1.0559 → 0.8599** and α=0.99 **1.3882 → 0.9099**. Comparing the last three reported windows (22.5k–30k) to the previous three (15k–22.5k), mean training CVaR improves **0.83%** (95%) and **1.42%** (99%). This supports *near-plateau behavior*, but particularly the 99%-policy may still be improving. Neither has fixed-validation measurements during the archived training, so genuine optimization convergence and overfitting cannot be established from these logs.

**Current assessment:** The 99%-policy is substantially flatter and shifted on both hedge legs, the discrepancy remains on plotted states visited by the simulator, BN2 has an elevated running-vs-current mean shift, and training appears to be approaching a plateau. None alone proves a faulty implementation. Before retraining, inspect the new spatially stratified error table, original saved visual paths if available, and optionally a controlled BN recalibration vs original frozen BN using independent evaluation samples.

To reproduce the new independent 40k diagnostic:

```bash
python -m scripts.a2.diagnose_e2cs_support_bn --n-paths 40000 --simulation-seed 928371
```

To compute state support from the **original** 200k visualization states when `pnl.npz` exists:

```bash
python -m scripts.a2.diagnose_e2cs_support_bn --pnl results/e2cs/0.95/0/pnl.npz
```

### Regional decomposition of the k=15 errors (validated extension)

The extended checkpoint-and-fresh-path diagnostic in [Actions run 44259754846](https://github.com/wtwtheric/SDSC4107-Group-project-A2/actions/runs/44259754846) passed tests and generated the regional-error artifact. The **40,000** sampled states include an exact-zero variance atom of **55.885%**. **67.683%** have `V<0.02`, below the Figure 8–9 variance grid, while **9.695%** have `V>0.14`. This explains why the displayed grid jointly covers only **16.148%** of states. A main research limitation is therefore that plots over `V ∈ [0.02, 0.14]` largely omit the most frequently visited zero/near-zero variance regime under the current high-vol-of-vol QE-M simulator.

The region `S∈[99,101], V∈[0.02,0.04]` contains **611 paths** (1.528% of all 40k, 9.46% of in-grid states). Its stock RMSE is **0.0647** for alpha=0.95 and **0.1802** for alpha=0.99. It accounts for approximately **28.6%** and **50.3%**, respectively, of the *total squared stock delta error among the in-grid states*. Therefore the α=0.99 mismatch is disproportionately concentrated around the model call-spread peak (S near 100, low but nonzero V), even under probability weighting within the grid. This zone does not alone explain most **unconditional** pathwise hedge error because 83.85% of sampled states are not included.

For the narrower model stock peak neighborhood (`S∈[99.3,100.3], V∈[0.02,0.04]`), only **296 paths** (0.74% of all) are sampled. α=0.99 stock RMSE there is **0.2078** versus **0.0691** under α=0.95; it contributes approximately **32.4%** of in-grid squared stock error for α=0.99. The shifted NN peak zone (`S∈[96.3,97.3], V∈[0.02,0.04]`) contains **169 paths** (0.42% of all); α=0.99 stock RMSE is **0.0642**, contributing only **1.8%** of in-grid squared stock error.

This is a stronger location-specific conclusion than the original uniform-grid RMSE: the observed discrepancy is not uniformly spread across the plotted states, and it is particularly large near the **model** peak. It does not prove that the α=0.99 policy is poorly optimized for CVaR in the full distribution.

The full regional `n_paths`, conditional RMSE, signed bias and share of in-grid squared error for the two hedge legs are contained in `reports/e2cs/e2cs_support_bn_training.json` in the [Actions artifact](https://github.com/wtwtheric/SDSC4107-Group-project-A2/actions/runs/44259754846).

## Status and next evaluation

The qualitative pattern previously observed in the original figures is that alpha=0.99 has a much flatter stock surface than the model hedge. That **visual interpretation** is not itself a measured numerical diagnosis. The numerical diagnostics should be run and inspected before attributing the flattening to input normalization, effective tail sample size, BatchNorm statistics, or policy optimization. Those are competing hypotheses, not demonstrated causes.

The seed-0 diagnosis has now been reviewed and the frozen recipe has been run for seeds 1–4. Do **not** modify the architecture, loss function, or training config or launch further E2cs retraining solely to match the paper figures. The consolidated five-seed result is documented in [e2cs_multiseed_summary.md](e2cs_multiseed_summary.md).

## Original 200K visual-path audit and BN follow-up (2026-11-18)

Two separately uploaded NPZ files, e2cs_095_pnl.npz and e2cs_099_pnl.npz, contain identical k=15 visual_spot and visual_variance arrays (200,000 states), allowing paired comparisons without resimulation. The uploaded NPZ files are not committed to GitHub.

| State subset | Paths | Stock mean alpha 0.95 | Stock mean alpha 0.99 | Variance-swap mean alpha 0.95 | Variance-swap mean alpha 0.99 |
|---|---:|---:|---:|---:|---:|
| All | 200,000 | 0.12072 | 0.03876 | 45.47 | -0.57 |
| V=0 | 110,950 | 0.15170 | 0.03538 | 50.44 | -7.83 |
| V=0 and 99 <= S <= 101 | 33,708 | 0.34599 | 0.04650 | 283.93 | -7.31 |
| Figure grid S in [95,103], V in [0.02,0.14] | 32,588 | 0.09388 | 0.04865 | 35.03 | 5.84 |

The strong near-strike, zero-variance position gap is present on 16.85% of the original 200K visual paths. This does not establish a failure of tail-risk optimization; it does establish that the difference is not an artifact of sparse Figure 8/9 grid points.

Current E2cs simpler input features are raw (S_k, V_k). In the original visual sample the spot standard deviation is 3.955 and variance standard deviation is 0.0796 (approximately 49.7 times smaller). First-layer BatchNorm normalizes activations, but cannot guarantee equal learning sensitivity to both raw inputs. This is a testable optimization concern, not proof of a feature bug.

Next experiment: python -m scripts.a2.diagnose_e2cs_original_bn --npz-95 /path/e2cs_095_pnl.npz --npz-99 /path/e2cs_099_pnl.npz

That script uses the existing, corrected simpler seed-0 checkpoints, checks that frozen-BN outputs reproduce the actual saved visual positions, compares frozen running moments against moments of the entire original 200K visual-state sample, and compares within-figure-grid model hedge errors under frozen BN versus non-mutating sample-moment BN.

Crucial limitations: sample-moment BN calculated from OOS data is a look-ahead counterfactual, not a valid deployed strategy or OOS CVaR result. The NPZ contains only k=15 market states plus terminal losses, not complete trajectories for re-evaluating all trades.

If a controlled retraining ablation is eventually warranted, retain the original raw-input seed-0 result. Test deterministic input scaling as a separately labeled E2cs-only implementation variant, while keeping two features, simpler architecture, market, hedge legs and CVaR objective. Changing YAML alone cannot change existing hard-coded features; the training inputs and all plot/diagnostic grid inputs must be changed consistently. Do not simultaneously modify BatchNorm, training batch, network width and simulator.


## Independent post-training BatchNorm calibration experiment (2026-11-18)

### Why this experiment is needed

The corrected alpha=0.99 simpler-policy checkpoint is architecturally valid and reproduces the archived k=15 visual positions. However, when its pre-BN activations are measured on independent Heston states, BN2's median standardized running-mean discrepancy is materially larger than for alpha=0.95. A non-mutating sample-moment counterfactual also changes the alpha=0.99 k=15 hedge positions substantially. This is evidence of inference-statistics sensitivity, not proof that the original checkpoint is incorrectly trained.

**This BatchNorm sensitivity is recorded as an implementation limitation of the E2cs seed-0 replication.** It may contribute to the visibly lower and left-shifted alpha=0.99 stock-hedge peak in Figure 9. It does not, by itself, explain the full difference from the model hedge or invalidate the separately measured OOS CVaR99 improvement.

### Controlled recalibration design

A separate branch, \`exp/e2cs-bn-calibration\`, contains a post-training recalibration ablation. It does not alter the original seed-0 checkpoints or original NPZ results.

For each of alpha=0.95 and alpha=0.99:

1. Load the original corrected simpler-policy checkpoint and its saved config.
2. Generate 200,000 new Heston QE-M calibration paths using calibration seed \`928372\`. These paths are distinct from the archived 200,000 visual paths and from the OOS evaluation paths.
3. For each of the 30 date-specific MLPs, estimate BN1's full-sample pre-BN activation mean and unbiased variance; install these as BN1 running buffers.
4. Pass the same calibration states through the now-calibrated BN1 and frozen FC2; estimate BN2's pre-BN moments and install BN2 running buffers.
5. Do not update Linear weights, BN affine parameters (\`weight\` / \`bias\`), input features, output scales, optimizer state, or training objective. No gradient step is performed.
6. Evaluate original and recalibrated policies on the same newly generated 1,000,000 Heston OOS paths using OOS seed \`314160\`. The calibration and OOS seeds are different. The paired paths ensure the comparison isolates the inference-buffer change.
7. Report mean loss, standard deviation, VaR, CVaR95, CVaR99, paired loss differences, worst-tail overlap, k=15 position changes/correlations, and stock/variance-swap surface peak diagnostics.

The calibration uses full-sample moments rather than an EMA over mini-batches. PyTorch's unbiased variance convention is used for \`running_var\`. BN1 is calibrated before BN2 because BN2's input distribution depends on BN1's inference transformation. The script records pre- and post-calibration moment discrepancies at every time step.

### Reproduction command

From the repository root, with the corrected seed-0 checkpoints available locally:

\`\`\`bash
python -m scripts.a2.run_e2cs_bn_recalibration \\
  --checkpoint-root backups/e2cs_simple_seed0/e2cs \\
  --calibration-paths 200000 \\
  --calibration-batch-size 25000 \\
  --calibration-seed 928372 \\
  --oos-paths 1000000 \\
  --oos-batch-size 25000 \\
  --oos-seed 314160 \\
  --device cuda \\
  --output results/e2cs_bn_calibration_seed0
\`\`\`

Use \`--device cpu\` if CUDA is unavailable. The original checkpoints are read-only inputs. Recalibrated checkpoint copies and the JSON report are written under the separate output directory. The report includes source and output SHA-256 hashes.

### Interpretation rules

- If recalibration moves the alpha=0.99 surface peak toward the original model peak and improves paired OOS CVaR without materially worsening mean loss, that supports BN running-statistics mismatch as a contributor. It does not establish that the original training objective or architecture was wrong.
- If the surface changes but OOS CVaR worsens, the original frozen checkpoint remains the primary replication result; the recalibrated variant is a sensitivity ablation only.
- If both hedge surfaces and OOS loss metrics barely change, BN mismatch is unlikely to be a material explanation for the Figure 9 discrepancy.
- Alpha=0.95 is a control. Large changes in both policies would suggest that the calibration procedure or sample definition needs review.
- The recalibrated model is not a replacement for the original model unless the group explicitly decides to report it as a separate implementation variant.

### Current status and limitation

The calibration helper, paired OOS experiment script, and unit tests have been added to this branch. **The full 200K-calibration / 1M-OOS experiment has not yet been run.** Therefore, no recalibration improvement or deterioration is claimed at this stage. The existing BN sensitivity evidence remains a limitation to disclose in the replication report, regardless of the outcome of this ablation.

This is a post-training normalization ablation, not retraining. Retraining is deferred until this controlled experiment is completed and reviewed.

## Independent BN recalibration result and freeze decision (2026-11-18)

The controlled post-training BatchNorm recalibration experiment has now been run on GPU with 200,000 independent calibration paths (seed 928372) and 1,000,000 independent paired OOS paths (seed 314160). Only BN running_mean/running_var buffers were changed; weights, BN affine parameters, features, output scaling, market dynamics and objective were frozen.

### OOS loss results

| Policy | Metric | Original | Recalibrated | Change |
|---|---|---:|---:|---:|
| alpha=.95 | mean loss | 0.609060 | 0.609145 | +0.014% |
| alpha=.95 | CVaR95 | 0.957418 | 0.856434 | -10.55% |
| alpha=.95 | CVaR99 | 1.084239 | 0.920275 | -15.12% |
| alpha=.99 | mean loss | 0.608900 | 0.608973 | +0.012% |
| alpha=.99 | CVaR95 | 0.922460 | 0.887164 | -3.83% |
| alpha=.99 | CVaR99 | 0.962205 | 0.907916 | -5.64% |

The mean loss is essentially unchanged while tail risk falls materially for both policies. This indicates a systematic train/eval BatchNorm-statistics effect rather than a failed alpha=.99 weight solution alone.

### Figure-9 / hedge-surface effect

For alpha=.99 the stock global peak changes from 0.11663 at S=96.8 to 0.16443 at S=98.7. At V=.03 the peak changes from 0.10752 at S=96.5 to 0.15668 at S=98.4. In contrast, alpha=.95 changes only modestly: global peak 0.32417 at S=99.4 to 0.29877 at S=99.7; at V=.03, 0.24983 at S=99.0 to 0.24308 at S=99.5.

Thus BN recalibration removes part of the unusually low/left-shifted alpha=.99 peak, but does not eliminate the economic risk-aversion pattern. After recalibration, at V=.03 the alpha=.99 peak remains only ~64.5% of the alpha=.95 peak and remains about 1.1 spot units to the left. This remains qualitatively consistent with the paper's statement that higher risk aversion flattens the call-spread hedge and behaves like a downward barrier/strike shift.

At k=15, alpha=.95 original/recalibrated positions remain highly correlated (stock 0.9875, variance swap 0.9947), while alpha=.99 is substantially more sensitive (stock 0.7474, variance swap 0.7199). This supports reporting BN inference sensitivity as an implementation limitation.

### Replication interpretation

The paper explicitly uses BatchNorm before each activation, Adam with learning rate .005 and batch size 256. Our A1/A2 stabilized recipe keeps the same two-hidden-layer 17-unit architecture and .005 initial learning rate, but uses batch 1024 and cosine decay for continuity with the validated A1 implementation. The paper does not specify a post-training running-statistics recalibration procedure. Therefore the recalibrated checkpoint should be reported as a diagnostic / implementation ablation rather than silently replacing the primary seed-0 replication result.

### Decision

Freeze the existing E2cs training weights, features, architecture, Heston parameters and A1-compatible training recipe. Do not retrain solely to make Figure 9 visually closer to the paper. The current evidence shows that a material part of the discrepancy is attributable to BatchNorm inference statistics and can be reduced without changing learned weights, while the economically meaningful high-alpha flattening remains.

A paper-fidelity retrain with batch 256 / constant lr=.005 may be run only as an optional sensitivity experiment if compute/time permits. It must be labeled separately because it intentionally departs from the A1-stabilized recipe and changes two optimization hyperparameters at once. It is not required to accept the current E2cs replication as a comprehensive but non-exact reproduction.


## Multi-seed completion (2026-11-19)

Corrected E2cs seeds 0–4 are complete under the frozen recipe. The alpha=.99 policy lowers both CVaR95 and CVaR99 relative to the alpha=.95 policy in all five seeds. The primary alpha=.99 CVaR99 is 0.991589 ± 0.052823, while CVaR99 under the alpha=.95 policy is 1.124302 ± 0.070055; the paired reduction is 0.132714 ± 0.033466. Seed 3 is the main numerical outlier, but retains the same direction of improvement.

Independent BN recalibration across all five seeds sharply reduces cross-seed dispersion while preserving the flatter and left-shifted alpha=.99 hedge. This strengthens the seed-0 conclusion that BatchNorm inference statistics are a material implementation sensitivity, but the original checkpoint evaluations remain the formal replication outputs. See [E2cs five-seed robustness summary](e2cs_multiseed_summary.md) and `reports/e2cs/e2cs_multiseed_summary.json` for the consolidated tables and machine-readable aggregates.
