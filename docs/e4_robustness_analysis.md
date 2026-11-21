# E4 Five-Seed Robustness Analysis — Frozen

Status: **frozen**. Seeds 0–4 are complete for both Black–Scholes (E4-BS) and Heston (E4-H). No seed is excluded and no further retraining is planned.

## 1. Robustness protocol

For each market, five independently trained seeds (0–4) are evaluated at the same five transaction-cost levels, epsilon = 2^-10,...,2^-6. Each trained policy is evaluated on 100,000 OOS paths using the common evaluation seed 271828. A separate log-log regression is fitted within each training seed:

```
log(p_epsilon - p0) = a + b log(epsilon).
```

Cross-seed uncertainty is summarized over the five fitted slopes. The 25 observations are **not pooled** as if they represented 25 independent epsilon levels.

Seed 0 was trained at commit feb6c7b and seeds 1–4 record de1ddfc, but the E4 implementation, E4-BS/E4-H configs, common metrics, network implementation and Heston environment are byte-identical across these commits.

## 2. Full-grid slopes

| Model | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± sample SD | Median |
|---|---:|---:|---:|---:|---:|---:|---:|
| Black–Scholes | 0.7111 | 0.7125 | 0.7158 | 0.7127 | 0.7123 | **0.7129 ± 0.0017** | 0.7125 |
| Heston | 0.6534 | 0.6562 | 0.6545 | **0.6160** | 0.6546 | **0.6469 ± 0.0173** | 0.6545 |

The theoretical continuous-time exponent is 2/3 ≈ 0.6667.

## 3. Black–Scholes robustness

BS is exceptionally stable across training seeds. The five fitted slopes span only 0.7111 to 0.7158, with sample SD 0.0017. Every full-grid fit has R² between 0.9990 and 0.9993.

The result supports a highly reproducible sublinear power-law relation over the finite epsilon grid. The exponent is systematically above 2/3, so the conclusion is approximate scaling rather than exact numerical recovery of the theoretical asymptotic exponent.

## 4. Heston robustness and seed 3

Heston seeds 0, 1, 2 and 4 cluster tightly around **0.6546 ± 0.0012**. Seed 3 gives the lower slope 0.6160, but it is not a failed run: all five price increments are positive and monotone in epsilon, all engineering checks pass, and the five-point log-log R² remains 0.9975.

The seed-3 difference is concentrated at low transaction costs:

| epsilon | seed-3 increment | mean of seeds 0,1,2,4 | seed 3 above others |
|---:|---:|---:|---:|
| 0.0009765625 | 0.254070 | 0.229857 | 10.53% |
| 0.001953125 | 0.374166 | 0.342277 | 9.32% |
| 0.00390625 | 0.556222 | 0.533590 | 4.24% |
| 0.0078125 | 0.866965 | 0.855176 | 1.38% |
| 0.015625 | 1.411269 | 1.405874 | 0.38% |

The relative deviation therefore increases as epsilon becomes small. This is consistent with greater optimization / finite-epsilon sensitivity in the two-hedge-instrument Heston problem. It should be disclosed as robustness uncertainty, not removed by excluding or retraining seed 3.

## 5. Final interpretation

**Black–Scholes:** strong cross-seed robustness. The five independently fitted slopes are essentially identical, so the observed finite-grid scaling is not dependent on a favorable training seed.

**Heston:** robust qualitative scaling with higher seed sensitivity. Every seed preserves a strong monotone, near-linear log-log relation; four seeds cluster near 0.655 and one seed (3) produces a lower slope because small-epsilon increments remain higher. The five-seed result is therefore reported as 0.6469 ± 0.0173, with median 0.6545.

This supports the paper's main E4 direction — transaction-cost premia grow approximately according to a two-thirds-type sublinear power law — while also showing that the more complex Heston case is more sensitive to optimization at low epsilon.

## 6. Freeze decision and limitations

E4 is frozen with all five seeds retained. No parameter tuning or retraining is performed after observing the robustness results.

Limitations to report:
- only five finite positive epsilon levels are used;
- the experiment uses 30-step discrete trading and finite 100K-path OOS estimates;
- p0 is the risk-neutral reference price rather than a separately optimized zero-cost discrete-time entropic price;
- LayerNorm and the A1/A2 stabilized optimization recipe are implementation deviations from the paper;
- Heston shows greater low-epsilon optimization sensitivity, most visibly in seed 3.

Primary robustness statistics must be one fitted slope per seed followed by mean ± **sample SD**. Do not pool all 25 epsilon/seed observations into a single regression.
