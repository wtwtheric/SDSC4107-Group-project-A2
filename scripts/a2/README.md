# A2 scripts

All A2-specific command-line tools live in this directory. Run module commands
from the repository root using the `python -m scripts.a2.<module>` form.

## Experiment runners

| Module | Purpose |
|---|---|
| `run_a2_seed0_batch` | Selectable seed-0 batch runner for E2 Call Spread, E3 and E4 |
| `run_e2cs` | Train and evaluate the E2 normalized call-spread policies |
| `run_e3` | Train, evaluate, finalize or plot the E3 recurrent/simpler comparison |
| `run_e4_grid` | Run the E4 epsilon grid for BS or Heston |

## Evaluation and figures

| Module | Purpose |
|---|---|
| `evaluate_e2` | Evaluate frozen E2 policies on fresh common paths |
| `make_figures_a2` | Unified entry point for A2 figures |
| `make_e2_figures` | E2 Figures 6-7 |
| `make_e2cs_figures` | E2 Call Spread Figures 8-9 |
| `make_e3_figures` | E3 Figure 5 |
| `make_e4_figures` | E4 figures and slope tables |

## Controlled diagnostics and robustness checks

| Module | Purpose |
|---|---|
| `audit_e2e3_simulators` | Compare QE-M and paper-style Heston simulators |
| `run_e2e3_fidelity_control` | Run the final E2/E3 simulator and input controls |
| `retrain_e2_alpha99_fractional` | Controlled E2 alpha=.99 fractional-CVaR retraining |
| `compare_e2_cvar_estimators` | Compare frozen ceil- and fractional-CVaR policies |
| `diagnose_e2cs` | Rebuild and inspect E2 Call Spread hedge surfaces |
| `diagnose_e2cs_support_bn` | Independent-path support and BatchNorm diagnostic |
| `diagnose_e2cs_original_bn` | Audit original saved E2 Call Spread arrays |
| `run_e2cs_bn_recalibration` | Post-training BatchNorm recalibration ablation |

These diagnostics are retained because the final methodology or CI refers to
them. They are different from the historical one-off probes under
`scripts/diagnostics/`.

## Utility

| Module | Purpose |
|---|---|
| `archive_a2_seed0` | Package selected seed-0 outputs and figures for transfer |

## Common commands

```bash
python -m scripts.a2.run_a2_seed0_batch --seed 0 --device cuda
python -m scripts.a2.make_figures_a2 --only all
python -m scripts.a2.run_e3 --finalize-only --run-dir results/e3/0
```

Generated outputs belong under `results/`, selected reproducibility artifacts
under `backups/`, and canonical report figures under `figures/`.
