# Fractional empirical CVaR: numerical test report

**Date:** 2026-11-15  
**Environment:** PyTorch 2.10.0+cpu, CPU, one Torch thread.  
**Scope:** Compare the current ceiling-based `metrics.cvar` calculation with fractional boundary weighting. No production code was changed and no model was retrained.

## 1. Why test this

The current implementation in `src/common/metrics.py` calculates

\[
\widehat{\mathrm{CVaR}}_{\alpha}^{ceil}
=\frac{1}{\lceil(1-\alpha)N\rceil}
\sum_{j=1}^{\lceil(1-\alpha)N\rceil}L_{(j)},
\]

where losses are sorted from largest to smallest.

The empirical-quantile integral gives a fractional-boundary estimator. Let

\[
m=(1-\alpha)N,\quad k=\lfloor m\rfloor,\quad r=m-k.
\]

Then

\[
\widehat{\mathrm{CVaR}}_{\alpha}^{frac}
=\frac{\sum_{j=1}^{k}L_{(j)}+rL_{(k+1)}}{m}.
\]

The first \(k\) losses receive full weight; the next loss receives only weight \(r\).

## 2. Exact E2 training-batch arithmetic

For the current E2 batch size \(N=1,024\) and \(\alpha=0.99\):

| Quantity | Value |
|---|---:|
| Requested tail mass \((1-\alpha)N\) | 10.24 |
| Current ceiling count | 11 observations |
| Fractional full-weight observations | 10 observations |
| Fractional weight on observation 11 | 0.24 |
| Weight per observation under current estimator | 1/11 = 9.0909% |
| Weight per top-10 observation under fractional estimator | 1/10.24 = 9.7656% |
| Weight on the 11th observation under fractional estimator | 0.24/10.24 = 2.3438% |

Thus the change is not merely a different denominator. It reallocates gradient weight away from the boundary observation and towards the worst 10 observations.

### Deterministic ordered sample

For a strictly descending sample `1024, 1023, ..., 1`:

| Estimator | Result |
|---|---:|
| Current ceiling top-11 mean | 1019.000000 |
| Fractional top-10 + 0.24 × 11th, divided by 10.24 | 1019.371094 |
| Difference | +0.371094 (+0.0364%) |

This confirms the direction: for a descending loss sample, fractional CVaR is no smaller than the ceiling estimator. The ceiling estimator overweights the lower boundary observation relative to the exact 1% empirical tail.

## 3. Repeated synthetic minibatch experiment

A fixed seed (`20261115`) was used to generate 5,000 independent batches, each containing 1,024 losses. Two illustrative distributions were tested:

1. Standard normal losses.
2. A positively skewed / heavy-tailed mixture: `0.7 * Normal(0,1) + J`, where `J` is zero with probability 98.5% and otherwise a positive exponential jump with rate 1.5.

These are diagnostic distributions only. They are **not fitted to, or sampled from, the E2 Heston training process**.

### Standard normal batches

| Statistic | Current ceiling | Fractional |
|---|---:|---:|
| Mean CVaR99 across batches | 2.62694 | 2.64999 |
| Median | 2.62244 | 2.64533 |
| Batch-to-batch standard deviation | 0.13961 | 0.14362 |
| Mean fractional-minus-ceiling gap | — | +0.02305 |
| Standard deviation of the gap | — | 0.00680 |
| 5th–95th percentile of the gap | — | 0.01303 to 0.03530 |
| Mean relative increase | — | +0.864% |
| Gap / current-estimator batch SD | — | 0.165 |

### Positively skewed / heavy-tailed batches

| Statistic | Current ceiling | Fractional |
|---|---:|---:|
| Mean CVaR99 across batches | 1.96630 | 1.98788 |
| Median | 1.95489 | 1.97449 |
| Batch-to-batch standard deviation | 0.14698 | 0.15352 |
| Mean fractional-minus-ceiling gap | — | +0.02158 |
| Standard deviation of the gap | — | 0.00819 |
| 5th–95th percentile of the gap | — | 0.01082 to 0.03703 |
| Mean relative increase | — | +1.067% |
| Gap / current-estimator batch SD | — | 0.147 |

Across both examples, the scalar CVaR difference is around 0.02 (about 0.9–1.1%), smaller than the batch-to-batch standard deviation of the estimated CVaR (about 0.14–0.16). This supports the view that the scalar shift is modest relative to minibatch noise. It does **not** mean the estimator choice is irrelevant: the gradient weights on the 11 tail observations change substantially, which can alter the optimisation path over many iterations.

## 4. Cases where the estimators are exactly equal

If \((1-\alpha)N\) is an integer, fractional weighting has no boundary fraction and both estimators agree.

| Use | \(N\) | \(\alpha\) | Tail mass | Difference? |
|---|---:|---:|---:|---|
| E1/E2 50%-CVaR training batch | 1,024 | 0.50 | 512 | No |
| E2 99%-CVaR evaluation (current) | 100,000 | 0.99 | 1,000 | No |
| E2 99%-CVaR final evaluation (planned) | 1,000,000 | 0.99 | 10,000 | No |
| 95%-CVaR evaluation | 100,000 | 0.95 | 5,000 | No |

Therefore:

- The current E2 `alpha=0.50` training is unaffected at batch size 1,024.
- The E2 `alpha=0.99` training objective is affected at batch size 1,024.
- Moving the evaluation set from 100k to 1M improves tail-sample precision, but does not change the ceiling-versus-fractional estimator in this case because both tail counts are integers.
- For a 1,024-path batch, `alpha=0.95` also has a non-integer tail mass (51.2), so the same issue applies to any E3/call-spread training using that combination.

## 5. What this does and does not explain about E2 seed 0

The first E2 run reports `price_alpha99 = 2.6619`, below the paper's approximately 3.49. The synthetic diagnostic suggests that the ceiling approximation can depress the minibatch CVaR objective by around 0.02 in the tested distributions, but it does not establish the size of the effect on the actual Heston training batches.

In particular, this estimator difference alone is not a credible explanation for the full 0.83 price gap. The real effect is on training gradients and therefore must be measured by retraining the same seed with the fractional estimator. The repository snapshot did not contain saved per-iteration terminal-loss minibatches, so a direct re-evaluation on the actual E2 training batches was not possible from the available artifacts.

## 6. Recommendation

1. Adopt fractional boundary weighting as the production empirical-CVaR definition for future CVaR training. It is the direct finite-sample integral of the empirical quantile function and avoids giving the boundary observation a full observation's weight when only a fraction of the tail mass is required.
2. Change the shared implementation once, in `src/common/metrics.py`, rather than implementing separate estimators in E2/E3.
3. Add focused unit tests for integer tail mass, fractional tail mass, cash translation invariance, and gradients at the boundary.
4. Retrain the E2 `alpha=0.99` seed-0 policy as a controlled comparison before starting seeds 1–4. The `alpha=0.50` policy does not need retraining for this change at batch size 1,024.
5. Use the same fractional estimator for future E3/CVaR training where the tail mass is non-integer (e.g. `alpha=0.95`, batch 1,024). Evaluation at 100k/1M paths is numerically unchanged for the listed confidence levels.

This recommendation is about consistency with the mathematical empirical-CVaR definition. It does not guarantee that the E2 price will move towards 3.49 or that the new training run will pass the paper's price band.

## 7. Reproduction

From the repository root:

```bash
python scripts/diagnostics/fractional_cvar_numerical_test.py
```

The script uses a small local mirror of the current ceiling-based calculation so it can run independently of the repository package import path. The source implementation was separately inspected in `src/common/metrics.py` on the repository's `main` branch; the mirror reproduces its validation, `ceil((1-alpha)N - 1e-12)`, and `torch.topk(...).mean()` logic.
