# E3 single-entry workflow

`scripts/run_e3.py` is the only E3 entry point. It supports both a full run and an evaluation-only run using saved checkpoints.

## Full run

From the repository root:

```bash
python -m scripts.run_e3 --seed 0 --device cuda
```

This trains recurrent and simpler policies, saves checkpoints and paired OOS terminal losses, estimates each indifference price on a separate calibration sample, evaluates OOS CVaR, and generates Figure 5.

## Re-evaluate existing checkpoints (no retraining)

```bash
python -m scripts.run_e3 \
  --eval-only \
  --run-dir results/e3/0 \
  --calibration-paths 1000000 \
  --calibration-batch-size 25000 \
  --calibration-seed 271828 \
  --device cuda
```

The run directory must contain `config.yaml`, `pnl.npz`, `policy_recurrent.pt`, `policy_simpler.pt`, and `metrics.json`. The original OOS loss arrays are reused; their seed is read from the saved config (`train.eval_seed`, 314159 in the current run). New calibration paths use a distinct seed (271828 by default). Both policies receive identical paths within each sample.

## Outputs

The run directory contains:

- `metrics.json`: original training provenance plus independent calibration/OOS metrics and checks.
- `pnl.npz`: original arrays plus calibration losses and centered OOS hedging errors.
- `evaluation.json`: sample counts, seeds, metric definitions, and independent-evaluation results.
- `fig_05_e3.png`: overlaid terminal hedging-error histogram and comparison table, saved at 300 dpi without an in-plot title.
- `evaluation.log`: created for evaluation-only mode; full runs log to `run.log`.

## Metric definitions

Let L be the terminal loss from the Heston environment, including proportional transaction costs, and alpha=0.99.

1. **Indifference price**: p0 = CVaR_alpha(L_cal), estimated on fresh calibration paths.
2. **Figure 5 terminal hedging error**: p0 - L_OOS.
3. **OOS realized CVaR shortfall**: CVaR_alpha(L_OOS - p0). Lower is better.
4. **Mean OOS loss**: arithmetic mean of uncentered L_OOS.

The calibration and OOS samples are independent by seed and shared between recurrent and simpler policies. This provides an independent-price/OOS-risk estimate but does not guarantee exact numerical agreement with the paper.

## Current replication status

The existing seed-0 result has recurrent price 4.1707 and simpler price 4.2261, below the plan's target bands [5.2, 5.8] and [6.4, 7.1]. The recurrent policy has lower OOS mean loss and OOS CVaR in this run, but the price gap is small. The result is therefore a valid evaluation run, not yet a quantitative reproduction of the paper's Figure 5 target values.

## Replication assessment framework

E3 is assessed at three distinct levels. A result can be methodologically useful even when its numerical values do not match the paper exactly; these levels should therefore be reported separately.

### 1. Implementation fidelity

Check whether the experiment implements the intended comparison and whether deviations from the paper are explicit and controlled.

- Both policies use the same Heston market, payoff, proportional transaction-cost formula, risk-aversion level (alpha=0.99), training budget, and base market information.
- The simpler policy receives the current market information I_k. The recurrent policy receives the same I_k plus the previous hedge position delta_{k-1}. This is the architectural distinction being tested; the two policies do not use different market feature definitions.
- Both policies use the same configured hidden width, activation, normalization choice, and output scaling. The current A2 configuration uses LayerNorm and the PFHedge QE-M Heston path generator. These are retained A1/A2 engineering choices and should be documented as deviations where they differ from the paper, rather than changed simultaneously during E3 diagnosis.
- Training uses on-the-fly Monte Carlo simulation. With 30,000 iterations and batch size 1,024, this corresponds to 30.72 million simulated training paths per policy. It is not a fixed, pre-generated dataset of 30.72 million paths.
- The independent calibration sample and OOS evaluation sample are each 1 million paths in the current seed-0 run. They are distinct from the training paths and use different random seeds. Within each sample, recurrent and simpler policies are evaluated on common paths.

### 2. Numerical replication

Compare the numerical outputs with the paper's Figure 5 and the pre-specified target ranges, while preserving the exact definition and sample used for each metric.

- Compare the indifference prices with the paper's reported values (approximately 5.51 for recurrent and 6.74 for simpler) and the project target bands.
- Report the recurrent-minus-simpler price difference and relative difference, not only whether each price falls inside a target band.
- Report the OOS mean loss and OOS realized CVaR shortfall with their definitions. The mean OOS loss in this implementation is the arithmetic mean of the uncentered terminal loss L_OOS; it should not automatically be treated as identical to a paper table entry unless the paper's centering and averaging convention is confirmed.
- Show the terminal hedging-error distributions p0-L_OOS for both policies, using the same histogram bins and independent calibration procedure.
- Treat seed 0 as the initial end-to-end replication run. Robustness across seeds 1-4 is a subsequent check, not a requirement for declaring the seed-0 workflow operational.

### 3. Economic interpretation

Explain what the comparison says about hedging performance under transaction costs, without claiming more than the metrics support.

- A lower indifference price means less initial capital is required under the chosen risk measure and calibration procedure.
- A lower OOS mean loss indicates a lower average terminal loss on the evaluation sample.
- A lower OOS realized CVaR shortfall indicates a more conservative realized tail outcome relative to the independently calibrated price; values near zero indicate that calibration and OOS tail risk are closely aligned.
- Use turnover and transaction-cost breakdowns (total, stock leg, and variance-swap leg) to investigate the mechanism. A lower price alone does not establish that the recurrent policy trades less; the cost and turnover metrics are needed to support that explanation.
- If the recurrent and simpler results are close, report that the recurrent advantage is small in this run. Do not force a strong architectural conclusion from one seed.

These three levels support a transparent replication report: implementation fidelity explains what was reproduced, numerical replication quantifies agreement with Figure 5, and economic interpretation explains the observed differences and their limits.
