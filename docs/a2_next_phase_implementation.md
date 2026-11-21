# A2 Next-Phase Implementation Package

> Baseline inspected: `main` at `c9c57f97dcfdee99cffe06fbf12477c702f68424` (2026-11-15).
> This package adds isolated runners and does not overwrite A1/E1/E2 source files.

## Scope

This package prepares and runs Seed 0 for:

1. E2 call spread: normalized payoff `((S_T-K1)+-(S_T-K2)+)/(K2-K1)`, `K1=100`, `K2=101`, `alpha=.95/.99`, zero cost.
2. E3: `eps=.01`, `alpha=.99`, recurrent versus simpler policies.
3. E4-BS and E4-Heston: entropic risk `lambda=1`, epsilon grid `[2^-6, 2^-7, 2^-8, 2^-9, 2^-10]`.

E1/E2's existing European-call payoff, QE-M simulator, shared fractional CVaR, and network implementation are reused without modifying their source files. E3/E4 training runners are separate so E1's existing checks and CVaR objective are not silently changed.

## New files

- `src/replication/e2cs.py`: call-spread world subclass, paired alpha training, chunked common-path OOS evaluation, model hedge grid.
- `src/replication/e3.py`: paired recurrent/simpler training, common OOS paths, instrument-level transaction-cost breakdown.
- `src/replication/e4.py`: entropic training for BS d=1 and Heston d=2.
- `scripts/a2/run_e2cs.py`: writes `results/e2cs/0.95/<seed>/` and `results/e2cs/0.99/<seed>/`.
- `scripts/a2/run_e3.py`: writes `results/e3/<seed>/`.
- `scripts/a2/run_e4_grid.py`: writes `results/e4bs/<eps>/<seed>/` and `results/e4h/<eps>/<seed>/`.
- `scripts/a2/run_a2_seed0_batch.py`: runs E2CS, E3, E4-BS, E4-Heston sequentially on one GPU.
- `scripts/a2/make_e2cs_figures.py`: makes 3D surfaces and 2D heatmaps.
- `scripts/a2/archive_a2_seed0.py`: bundles the completed Seed-0 result folders and figures into one ZIP.
- Configs and small runner tests under `configs/exp/` and `tests/`.

## Run locally or on AutoDL/JupyterLab

From the repository root after overlaying this package:

```bash
python -m pytest tests/test_e2cs.py tests/test_e3.py tests/test_e4.py -q
python -m ruff check src/replication/e2cs.py src/replication/e3.py src/replication/e4.py scripts/a2/run_e2cs.py scripts/a2/run_e3.py scripts/a2/run_e4_grid.py scripts/a2/run_a2_seed0_batch.py scripts/a2/make_e2cs_figures.py scripts/a2/archive_a2_seed0.py tests/test_e2cs.py tests/test_e3.py tests/test_e4.py
python -m ruff format --check src/replication/e2cs.py src/replication/e3.py src/replication/e4.py scripts/a2/run_e2cs.py scripts/a2/run_e3.py scripts/a2/run_e4_grid.py scripts/a2/run_a2_seed0_batch.py scripts/a2/make_e2cs_figures.py scripts/a2/archive_a2_seed0.py tests/test_e2cs.py tests/test_e3.py tests/test_e4.py
python -m scripts.a2.run_a2_seed0_batch --seed 0 --device cuda
# after the full batch, one archive is created at results/A2_seed_0_bundle.zip
```

To run one stage only:

```bash
python -m scripts.a2.run_e2cs --seed 0 --device cuda
python -m scripts.a2.run_e3 --seed 0 --device cuda
python -m scripts.a2.run_e4_grid --config configs/exp/e4bs.yaml --seed 0 --device cuda
python -m scripts.a2.run_e4_grid --config configs/exp/e4h.yaml --seed 0 --device cuda
python -m scripts.a2.make_e2cs_figures --seed 0
```

The batch script is sequential, not parallel. It runs the E2CS figure generation immediately after E2CS, then E3, E4-BS, and E4-Heston. It uses `--resume` by default and skips completed runs with an existing `metrics.json`. A partial E2CS pair is treated as an error rather than silently mixing runs. At the end, it creates one archive at `results/A2_seed_0_bundle.zip`. Use a fresh output directory (`--out results_seed0_retry`) for a clean rerun.

## Result structure

```text
results/
  e2cs/
    0.95/0/{metrics.json,config.yaml,run.log,pnl.npz,policy.pt}
    0.99/0/{metrics.json,config.yaml,run.log,pnl.npz,policy.pt}
    seed_0_summary.json
  e3/0/{metrics.json,config.yaml,run.log,pnl.npz,policy_recurrent.pt,policy_simpler.pt}
  e4bs/<epsilon>/0/{metrics.json,config.yaml,run.log,pnl.npz,policy.pt}
  e4h/<epsilon>/0/{metrics.json,config.yaml,run.log,pnl.npz,policy.pt}
  e4bs/seed_0_epsilon_summary.json
  e4h/seed_0_epsilon_summary.json
  A2_seed_0_bundle.zip
figures/
  fig08_e2cs_stock_position.png
  fig09_e2cs_varswap_position.png
  fig08_e2cs_stock_position_heatmap.png
  fig09_e2cs_varswap_position_heatmap.png
```

## Important interpretation notes

- E2CS uses the normalized call spread exactly as specified in `docs/plan_A2_next.md`. The original European-call default in `HestonWorld` is untouched.
- The policy is semi-recurrent. The plotted policy surfaces are conditional medians of positions observed after full paths have passed through the policy; they are not obtained by feeding an isolated `(S,V)` state with a zeroed recurrent position.
- The E2CS model delta grid is calculated from the Heston call-price surface using central finite differences in spot and variance. It is a numerical benchmark; grid resolution and finite-difference steps are saved in source and should be sensitivity-checked before final reporting.
- E3 reports total transaction costs and separate stock/variance-swap cost contributions. The zero-cost model price is contextual only; it is not an E3 acceptance check.
- E4's `positive_price_increment` check may fail for a small epsilon if the estimated premium is below numerical/training noise. Do not add an arbitrary offset before log regression. The final slope is assessed only after all five seeds and positivity/noise checks.
- E4 uses a dedicated entropic objective and does not alter the shared E1/E2 CVaR trainer.
- The default E2CS/E3 evaluation is 1,000,000 common paths in 25,000-path chunks. E4 uses 100,000 OOS paths per epsilon as the plan specifies.
- This package has not run a GPU training session. Python syntax compilation passed. Seven short runner tests also passed against temporary interface-compatible stubs, which checks the runner wiring only; it does **not** validate QE-M or the real repository integration. Run the tests again in the actual repository environment with the pinned `pfhedge` dependency before launching the full Seed-0 batch.
