# E3 independent calibration, OOS evaluation, and Figure 5

This workflow uses the frozen E3 seed-0 checkpoints and the paired 1,000,000-path
OOS terminal-loss arrays produced by the original training run. It does **not**
retrain either policy.

## Required source artifacts

The source directory must contain:

- `config.yaml`
- `metrics.json`
- `pnl.npz`, with `terminal_loss` (recurrent) and `terminal_loss_simpler`
- `policy_recurrent.pt`
- `policy_simpler.pt`

For the existing seed-0 run, this is normally `results/e3/0/`.

## Step 1 — independent calibration

Run from the repository root, on the same CUDA environment used for E3 if available:

```bash
python -m scripts.evaluate_e3 \
  --run-dir results/e3/0 \
  --calibration-paths 1000000 \
  --batch-size 25000 \
  --calibration-seed 314160 \
  --device cuda
```

The script simulates a fresh calibration sample with seed 314160. It resets the
same seed before evaluating each policy, so recurrent and simpler are priced on
the **same calibration paths**. The original OOS losses came from the E3 run's
evaluation seed (314159), so calibration and OOS samples are disjoint.

For each policy:

```text
p_cal = q + CVaR_alpha(L_cal - q)
H_oos = p_cal - L_oos
realized_CVaR = -CVaR_alpha(L_oos - p_cal)
```

The price is estimated only from the calibration paths. The original common OOS
loss arrays are reused unchanged for the out-of-sample histogram and realized
CVaR. The script writes:

- `results/e3/0/independent_evaluation/metrics.json`
- `results/e3/0/independent_evaluation/pnl.npz`

The NPZ retains both calibration loss arrays, both original OOS loss arrays,
and the two OOS hedging-error arrays.

## Step 2 — Figure 5

```bash
python -m scripts.make_e3_figure5 \
  --evaluation-dir results/e3/0/independent_evaluation \
  --figures-dir figures
```

Output: `figures/fig05_e3_recurrent_vs_simpler.png` (300 dpi).

The histogram shows `H = p_cal - L_oos`; the table reports mean absolute
hedging error `E[|H|]`, independently calibrated price, and the lower-tail
CVaR of `H`.

**Mean Loss caveat:** the Figure 5 caption in the paper does not define the
table's "Mean Loss" statistic. This implementation therefore labels its
transparent proxy as `Mean |H|` and does not claim that it is numerically
identical to the paper's Mean Loss column. Price and realized CVaR use the
explicit hedging-error definition in the paper caption.

## Reproducibility and interpretation

- Source checkpoint hashes and source PnL hash are written to `metrics.json`.
- Calibration seed, sample size, alpha, q, source commit, and OOS sample count
  are recorded.
- Both policies use common calibration paths and common OOS paths.
- No checkpoint, training config, or original result is overwritten.
- The script checks that the calibration seed differs from the original
  `train.eval_seed` (default 314159).
