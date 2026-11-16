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
