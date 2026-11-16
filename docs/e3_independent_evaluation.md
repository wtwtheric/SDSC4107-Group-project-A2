# E3 independent calibration and Figure 5

This workflow evaluates the already-trained E3 recurrent and simpler checkpoints. It does not retrain either policy and does not overwrite the original `metrics.json`, `pnl.npz`, or checkpoints.

## Run

From the repository root, with the original E3 seed-0 run available at `results/e3/0/`:

```bash
python -m scripts.evaluate_e3 \
  --run-dir results/e3/0 \
  --calibration-paths 1000000 \
  --calibration-batch-size 25000 \
  --calibration-seed 271828 \
  --device cuda
```

The original OOS losses are reused from `results/e3/0/pnl.npz`. Those paths were generated using `train.eval_seed=314159`. Fresh calibration paths are generated using seed 271828. Recurrent and simpler policies are evaluated on identical paths within each sample.

## Outputs

Written to `results/e3/0/independent_evaluation/`:

- `evaluation.json`: calibration and OOS metrics, sample counts, seeds, and definitions.
- `figure5_e3.png`: overlaid histogram of the terminal hedging error and a three-row-style comparison table.
- `evaluation_arrays.npz`: calibration losses, OOS losses, and centered OOS hedging errors for both policies.

## Metric definitions

Let (L) denote the terminal loss returned by the Heston environment, including transaction costs, and let (alpha=0.99).

1. **Indifference price**: (p_0=\mathrm{CVaR}_\alpha(L_{cal})), estimated on the fresh calibration sample. This is equivalent to (q+\mathrm{CVaR}_\alpha(L_{cal}-q)) by translation equivariance.
2. **Figure 5 terminal hedging error**: (p_0-L_{OOS}), matching the paper's (p_0-Z+(\delta\cdot S)_T-C_T(\delta)) sign convention.
3. **OOS realized CVaR shortfall**: (\mathrm{CVaR}_\alpha(L_{OOS}-p_0)). Lower values indicate a smaller upper-tail loss relative to the calibrated price.
4. **Mean OOS loss**: arithmetic mean of the uncentered (L_{OOS}). It is shown transparently and should not be confused with the paper's reported Mean Loss unless that table's exact normalization is separately reproduced.

The calibration and OOS samples are independent by seed and are shared between the two policies. The script reports the original OOS seed from the saved config for provenance.

## Important interpretation

This is a clean independent-price / OOS-risk evaluation using the current implementation and existing checkpoints. It does not by itself resolve the gap between this implementation and the paper's numerical Figure 5 values. In particular, the simulator, input features, normalization, and training batch size remain those of the saved E3 run.
