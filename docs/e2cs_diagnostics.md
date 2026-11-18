# E2cs corrected simpler-policy seed-0 diagnostics

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

`scripts/make_e2cs_figures.py` uses blue-white-red (`bwr`) rather than the grey-centered `coolwarm`. The same model/NN hedge color interval is used across Figures 8 and 9. Differences use another shared symmetric interval centered at **zero**, which remains white; blue indicates NN below the model and red indicates NN above it.

The model/NN hedge color scale is **not** forced to center on zero, avoiding a misleading grey/white block for small but meaningful stock positions. The 3D renderer uses `shade=False` to avoid lighting-driven color changes. Because the hedges are not signed differences, their color should not be interpreted as two sides of a risk sign.

The original figure assets remain untouched; diagnostics are written to `figures/e2cs_diagnostics/` until reviewed.

## Hedge-shape diagnostics

`scripts/diagnose_e2cs.py` reloads the frozen policy checkpoints with BatchNorm in evaluation mode and directly evaluates the same 81-by-61 uniform `(S,V)` grid used by the original Figures 8–9. It computes the analytical-model hedge using the existing Heston finite-difference pricer. It records **both** the stock and the variance-swap legs.

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
python -m scripts.diagnose_e2cs --device cpu
```

Or via the unified A2 entry point:

```bash
python -m scripts.make_figures_a2 --only e2cs-diagnostics --seed 0
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

A checkpoint-only and independent-path diagnostic has been implemented as `scripts/diagnose_e2cs_support_bn.py`, using the original simple-policy seed-0 weights; the same Heston model parameters and k=15. The initial successful run uses **40,000 newly generated CPU Heston paths, simulation seed 928371**, not the original GPU visual paths in the omitted `pnl.npz`. Results from this run are in GitHub Actions [run 44259116837](https://github.com/wtwtheric/SDSC4107-Group-project-A2/actions/runs/44259116837); the later regional-error extension is separately verified by CI before acceptance.

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
python -m scripts.diagnose_e2cs_support_bn --n-paths 40000 --simulation-seed 928371
```

To compute state support from the **original** 200k visualization states when `pnl.npz` exists:

```bash
python -m scripts.diagnose_e2cs_support_bn --pnl results/e2cs/0.95/0/pnl.npz
```

### Regional decomposition of the k=15 errors (validated extension)

The extended checkpoint-and-fresh-path diagnostic in [Actions run 44259754846](https://github.com/wtwtheric/SDSC4107-Group-project-A2/actions/runs/44259754846) passed tests and generated the regional-error artifact. The **40,000** sampled states include an exact-zero variance atom of **55.885%**. **67.683%** have `V<0.02`, below the Figure 8–9 variance grid, while **9.695%** have `V>0.14`. This explains why the displayed grid jointly covers only **16.148%** of states. A main research limitation is therefore that plots over `V ∈ [0.02, 0.14]` largely omit the most frequently visited zero/near-zero variance regime under the current high-vol-of-vol QE-M simulator.

The region `S∈[99,101], V∈[0.02,0.04]` contains **611 paths** (1.528% of all 40k, 9.46% of in-grid states). Its stock RMSE is **0.0647** for alpha=0.95 and **0.1802** for alpha=0.99. It accounts for approximately **28.6%** and **50.3%**, respectively, of the *total squared stock delta error among the in-grid states*. Therefore the α=0.99 mismatch is disproportionately concentrated around the model call-spread peak (S near 100, low but nonzero V), even under probability weighting within the grid. This zone does not alone explain most **unconditional** pathwise hedge error because 83.85% of sampled states are not included.

For the narrower model stock peak neighborhood (`S∈[99.3,100.3], V∈[0.02,0.04]`), only **296 paths** (0.74% of all) are sampled. α=0.99 stock RMSE there is **0.2078** versus **0.0691** under α=0.95; it contributes approximately **32.4%** of in-grid squared stock error for α=0.99. The shifted NN peak zone (`S∈[96.3,97.3], V∈[0.02,0.04]`) contains **169 paths** (0.42% of all); α=0.99 stock RMSE is **0.0642**, contributing only **1.8%** of in-grid squared stock error.

This is a stronger location-specific conclusion than the original uniform-grid RMSE: the observed discrepancy is not uniformly spread across the plotted states, and it is particularly large near the **model** peak. It does not prove that the α=0.99 policy is poorly optimized for CVaR in the full distribution.

The full regional `n_paths`, conditional RMSE, signed bias and share of in-grid squared error for the two hedge legs are contained in `reports/e2cs/e2cs_support_bn_training.json` in the [Actions artifact](https://github.com/wtwtheric/SDSC4107-Group-project-A2/actions/runs/44259754846).

## Status and next evaluation

The qualitative pattern previously observed in the original figures is that alpha=0.99 has a much flatter stock surface than the model hedge. That **visual interpretation** is not itself a measured numerical diagnosis. The numerical diagnostics should be run and inspected before attributing the flattening to input normalization, effective tail sample size, BatchNorm statistics, or policy optimization. Those are competing hypotheses, not demonstrated causes.

Do **not** modify the architecture, loss function, or training config at this stage, and do not launch multi-seed E2cs until the diagnosis is reviewed. E4 robustness training remains isolated in its own branch.
