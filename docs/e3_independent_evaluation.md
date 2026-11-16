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
- `pnl.npz`: original arrays plus calibration losses, fixed-validation losses, and centered OOS hedging errors.
- `evaluation.json`: sample counts, seeds, metric definitions, and independent-evaluation results.
- `fig_05_e3.png`: overlaid terminal hedging-error histogram and comparison table, saved at 300 dpi without an in-plot title.
- `evaluation.log`: created for evaluation-only mode; full runs log to `run.log`.

## Metric definitions

Let L be the terminal loss from the Heston environment, including proportional transaction costs, and alpha=0.99.

1. **Indifference price**: p0 = CVaR_alpha(L_cal), estimated on fresh calibration paths.
2. **Figure 5 terminal hedging error**: p0 - L_OOS.
3. **OOS realized CVaR shortfall**: CVaR_alpha(L_raw,OOS - p0). Lower is better.
4. **Raw mean loss**: E[L_raw] on the uncentered OOS sample. This is retained as a diagnostic and is not the Figure 5 table's Mean Loss.
5. **Signed terminal hedging loss (Figure 5 operational definition)**: define the paper-style terminal hedging loss as L_T = Z - p0 - (delta · S)_T + C_T(delta) = L_raw - p0. The reported Mean Loss is E[L_T] on the OOS sample. This follows the terminal-loss convention discussed for Figure 5; because the paper's Figure 5 caption does not separately spell out the table calculation, this operational interpretation is stated explicitly rather than presented as an independently verified formula.
6. **Positive-part excess loss diagnostic**: E[(L_raw - p0)^+] is also stored as an auxiliary diagnostic, not substituted for the paper's Mean Loss.
7. **Figure 5 histogram**: p0 - L_raw,OOS, matching the sign/order in the paper's caption.

The calibration and OOS samples are independent by seed and shared between recurrent and simpler policies. This provides an independent-price/OOS-risk estimate but does not guarantee exact numerical agreement with the paper.

## Current replication status

The existing seed-0 result has recurrent price 4.1707 and simpler price 4.2261, below the plan's target bands [5.2, 5.8] and [6.4, 7.1]. The uncentered OOS raw mean losses are 2.8704 and 2.9323, respectively. Under the signed terminal-loss convention above, the Figure 5 Mean Loss values are -1.3002 and -1.2938. The realized CVaR shortfalls are -0.0040 and -0.0027. The recurrent policy has a slightly lower price and raw mean loss, while the simpler policy has a slightly lower (more negative) realized CVaR shortfall. The price gap remains small. The result is therefore a valid evaluation run, not yet a quantitative reproduction of the paper's Figure 5 target values.

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


## Evaluation correction and training diagnostics

The evaluation code keeps the following quantities separate to prevent a change of convention from being hidden:

- `oos_mean_raw_loss`: E[L_raw], where L_raw = Z - (delta · S)_T + C_T(delta).
- `oos_signed_terminal_hedging_loss`: E[L_raw - p0]. This is the value displayed as Figure 5 "Mean Loss" and stored in the legacy `mean_loss_recurrent` / `mean_loss_simpler` metric fields.
- `oos_mean_excess_loss_diagnostic`: E[(L_raw - p0)^+], an auxiliary diagnostic only.
- `oos_realized_cvar_shortfall`: CVaR_alpha(L_raw - p0).
- `oos_mean_hedging_error`: E[p0 - L_raw], the signed mean of the histogram variable; it is the negative of the signed terminal hedging loss.

The Figure 5 table uses the original column labels: **Mean Loss**, **Price**, and **Realized CVaR**. The first is evaluated as the mean signed terminal hedging loss above. The paper's caption supplies the terminal hedging-error expression but does not independently define the averaging convention in the table; the project therefore records this interpretation transparently. Raw mean loss remains available in `evaluation.json` so readers can audit the centering.

### Fixed validation paths

The training function now supports an optional fixed validation sample. Set `train.validation_paths` to a positive count; `train.validation_seed` fixes the sample and `train.validation_batch_size` controls evaluation memory. At each existing `log_every` interval, it logs `validation_cvar` alongside the stochastic training CVaR. Validation generation uses a separate seed and restores CPU/CUDA RNG states, so enabling the diagnostic does not change the subsequent training-path random sequence.

The E3 config enables 10,000 fixed validation paths for future training runs. The evaluation-only workflow also evaluates the existing saved seed-0 checkpoints once on a separate 10,000-path fixed validation sample and stores both loss arrays and summary CVaR values. **The existing seed-0 checkpoints were not retrained**; the historical training log cannot be retroactively populated with validation values. The one-off final-checkpoint diagnostic is not a validation learning curve and cannot by itself establish the absence of overfitting.

### Current scope and limitations

- This correction is evaluation-only for the existing seed-0 models; model weights, architecture, features, Heston generator, optimizer, and training hyperparameters are unchanged.
- Seed 1-4 are intentionally deferred until the remaining A2 experiments have been reviewed.
- Numerical agreement with the paper is assessed separately from implementation fidelity and economic interpretation. The observed price gap remains a replication limitation, not a reason to alter the evaluation definition.
