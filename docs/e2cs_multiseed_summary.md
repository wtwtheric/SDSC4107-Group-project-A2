# E2cs five-seed robustness summary

## Scope

This document consolidates the corrected E2cs simpler-policy results for training seeds 0–4.

- Seed 0 primary checkpoints and metrics: `backups/e2cs_simple_seed0/e2cs/`
- Seeds 1–4 primary metrics and figures: `backups/e2cs_seeds1_4/`
- Seed 0 BN recalibration summary: `reports/e2cs/e2cs_bn_recalibration_seed0_summary.json`
- Seeds 1–4 BN recalibration reports: `backups/e2cs_seeds1_4/bn_recalibration/`

All five runs use the frozen A1-compatible E2cs recipe: the simpler per-date feed-forward policy, raw `(S_k,V_k)` inputs, BatchNorm, 30,000 iterations, batch 1,024, cosine learning-rate decay, and 1,000,000 common OOS paths per seed. The primary evaluation uses OOS seed 314159. The BN recalibration diagnostic uses an independent 200,000-path calibration sample (seed 928372) and a separate paired 1,000,000-path OOS sample (seed 314160).

The original checkpoints are the formal replication results. BN recalibration is a post-training sensitivity diagnostic and does not replace them.

## Primary five-seed results

Here `CVaR95(π95)` and `CVaR99(π99)` are the own-objective prices. The cross-risk columns evaluate both policies at the same risk level.

| Seed | CVaR95(π95) | CVaR99(π95) | CVaR95(π99) | CVaR99(π99) | CVaR99 reduction: π95 − π99 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.958340 | 1.087331 | 0.922781 | 0.963278 | 0.124053 |
| 1 | 1.004748 | 1.151164 | 0.920475 | 0.969055 | 0.182109 |
| 2 | 0.951869 | 1.073712 | 0.924386 | 0.968483 | 0.105229 |
| 3 | 0.998531 | 1.235711 | 0.990162 | 1.085939 | 0.149772 |
| 4 | 0.969135 | 1.073593 | 0.920687 | 0.971189 | 0.102405 |
| **Mean ± sample SD** | **0.976525 ± 0.023843** | **1.124302 ± 0.070055** | **0.935698 ± 0.030488** | **0.991589 ± 0.052823** | **0.132714 ± 0.033466** |

The alpha=.99 policy lowers CVaR99 relative to the alpha=.95 policy in **5/5 seeds**. It also lowers CVaR95 in **5/5 seeds**, by 0.040827 ± 0.028306. Mean terminal losses are extremely stable: 0.609580 ± 0.000079 for π95 and 0.609680 ± 0.000135 for π99, both within 0.001 of the model call-spread value `q=0.610462`. Thus the tail-risk differences are not driven by a material shift in the mean payoff.

Seed 3 is the main high-alpha variability case: `CVaR99(π99)=1.085939`, versus 0.963–0.971 in the other four seeds. It still improves on the same seed's `CVaR99(π95)=1.235711`. The qualitative risk-aversion result is therefore robust, while the numerical alpha=.99 price has non-negligible seed sensitivity.

## Independent BatchNorm recalibration across five seeds

The recalibration changes only BatchNorm `running_mean` and `running_var`; learned Linear weights, BN affine parameters, input features, output scales, market paths, and objectives remain frozen.

Because this diagnostic uses OOS seed 314160 rather than the primary seed 314159, its original-checkpoint values differ slightly from the table above.

| Policy | Metric | Original mean ± SD | Recalibrated mean ± SD | Mean percentage change |
|---|---|---:|---:|---:|
| π95 | CVaR95 | 0.976985 ± 0.024402 | 0.857103 ± 0.000398 | -12.23% |
| π95 | CVaR99 | 1.125127 ± 0.072103 | 0.922185 ± 0.001725 | -17.78% |
| π99 | CVaR95 | 0.935544 ± 0.030402 | 0.889165 ± 0.001239 | -4.88% |
| π99 | CVaR99 | 0.990870 ± 0.052578 | 0.908375 ± 0.000445 | -8.13% |

Recalibration reduces tail risk in every seed and sharply reduces cross-seed dispersion. This is consistent with the frozen BatchNorm running buffers being a material source of inference variability, including the seed-3 outlier. It does not prove that the learned weights are seed-invariant, and it does not make the recalibrated checkpoints the primary replication output.

## Hedge-surface robustness

At `V=0.03`, the five-seed stock-position peaks are:

| Policy | Original peak value | Original peak spot | Recalibrated peak value | Recalibrated peak spot |
|---|---:|---:|---:|---:|
| π95 | 0.205925 ± 0.030779 | 98.94 ± 0.63 | 0.234514 ± 0.011090 | 99.30 ± 0.16 |
| π99 | 0.134699 ± 0.021516 | 97.82 ± 1.66 | 0.145240 ± 0.008619 | 98.20 ± 0.34 |

After recalibration, the π99 peak is **0.620 ± 0.038** times the π95 peak and lies **1.10 ± 0.31 spot units to the left**. The higher-risk-aversion policy therefore remains flatter and left-shifted across seeds. Recalibration makes that qualitative pattern more consistent rather than removing it.

The unrecalibrated surface is less stable: the π99/π95 peak ratio is 0.666 ± 0.134 and the spot shift is -1.12 ± 1.45. Individual original checkpoints should not be over-interpreted from one seed.

## Replication conclusion

The five-seed experiment supports the paper's qualitative E2cs mechanism: stronger tail-risk aversion produces a flatter, left-shifted call-spread hedge and lowers extreme-tail loss without materially changing the mean loss. Exact prices and unrecalibrated surface locations remain sensitive to training seed and BatchNorm inference statistics.

For formal reporting:

1. Report the original five-seed mean ± sample SD as the replication result.
2. State that the CVaR99 improvement occurs in 5/5 seeds.
3. Identify seed 3 and the alpha=.99 price SD as evidence of numerical seed sensitivity.
4. Present BN recalibration only as a controlled diagnostic showing inference-statistics sensitivity.
5. Do not replace the original checkpoints or launch further E2cs retraining solely to match the paper's plots.

## Report-ready paragraph

> Across five independent training seeds, the alpha=.99 policy reduced CVaR99 by 0.1327 ± 0.0335 relative to the alpha=.95 policy on the same one-million-path OOS samples, and the improvement occurred in all five seeds. Mean losses remained within 0.001 of the model call-spread value, indicating that the difference was concentrated in the loss tail rather than the mean. The alpha=.99 price was more seed-sensitive (0.9916 ± 0.0528), mainly because of seed 3. A post-training BatchNorm recalibration substantially reduced cross-seed dispersion without changing learned weights, while the alpha=.99 hedge remained flatter and about 1.1 spot units left of the alpha=.95 hedge. We therefore retain the original checkpoints as the direct replication result and report BatchNorm sensitivity as an implementation limitation.
