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

## Status and next evaluation

The qualitative pattern previously observed in the original figures is that alpha=0.99 has a much flatter stock surface than the model hedge. That **visual interpretation** is not itself a measured numerical diagnosis. The numerical diagnostics should be run and inspected before attributing the flattening to input normalization, effective tail sample size, BatchNorm statistics, or policy optimization. Those are competing hypotheses, not demonstrated causes.

Do **not** modify the architecture, loss function, or training config at this stage, and do not launch multi-seed E2cs until the diagnosis is reviewed. E4 robustness training remains isolated in its own branch.
