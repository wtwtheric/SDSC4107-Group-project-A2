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

## Status and next evaluation

The qualitative pattern previously observed in the original figures is that alpha=0.99 has a much flatter stock surface than the model hedge. That **visual interpretation** is not itself a measured numerical diagnosis. The numerical diagnostics should be run and inspected before attributing the flattening to input normalization, effective tail sample size, BatchNorm statistics, or policy optimization. Those are competing hypotheses, not demonstrated causes.

Do **not** modify the architecture, loss function, or training config at this stage, and do not launch multi-seed E2cs until the diagnosis is reviewed. E4 robustness training remains isolated in its own branch.
