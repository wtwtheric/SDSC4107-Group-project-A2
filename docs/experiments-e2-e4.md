# A2 E2–E4 Experiment Registry

> **Status:** Frozen on 2026-11-21  
> **Scope:** A2 experiments E2–E4, corresponding to paper Figures 5–11  
> **Ownership:** This is the A2 registry. The A1 E1/Figures 1–3 registry remains in `docs/experiments.md` and is not modified here.

## 1. Freeze rule

“Frozen” means that the planned seeds, evaluations and robustness checks are complete and no further training will be performed solely to move the outputs closer to the paper. It does not mean that every paper-target acceptance band was passed. Quantitative discrepancies and provenance limitations remain part of the result.

Formal training uses seeds 0–4. E2, E2cs and E3 use one-million-path out-of-sample evaluation where specified. E4 evaluates each trained policy on 100,000 common OOS paths at each transaction-cost level.

## 2. Experiment summary

| Experiment | Formal design | Source commit(s) | Main result | Final status |
|---|---|---|---|---|
| E2 core, Figures 6–7 | Heston d=2; alpha=.50 vs .99; seeds 0–4; 30k iterations; 1M OOS paths per seed | `85bdcbc` | Price(alpha=.50) = 1.9432 ± 0.0007; price(alpha=.99) = 2.6595 ± 0.0053. The cross-risk ordering is stable, but the alpha=.99 price is below the paper-target band [3.2, 3.8]. | **Frozen — qualitative/partial; paper price target not passed** |
| E2 call spread, Figures 8–9 | K1=100, K2=101; alpha=.95 vs .99; simpler policy; seeds 0–4; 1M common OOS paths | seed 0: `1e75b7d`; seeds 1–4: `bfd7203` | The alpha=.99 policy lowers CVaR99 relative to the alpha=.95 policy in 5/5 seeds by 0.1327 ± 0.0335. Hedge surfaces remain flatter and left-shifted, with seed and BatchNorm sensitivity. | **Frozen — qualitative mechanism reproduced** |
| E3, Figure 5 | Heston d=2; epsilon=.01; alpha=.99; recurrent vs simpler; seeds 0–4; independent calibration and 1M OOS paths | seed 0: `feb6c7b`; seeds 1–4: `f65e886` | Recurrent price = 4.1646 ± 0.0056; simpler price = 4.2225 ± 0.0040. Recurrent is lower in 5/5 seeds, but the 0.0579 gap is much smaller than the paper gap 1.2309. | **Frozen — stable qualitative, incomplete quantitative reproduction** |
| E2/E3 fidelity controls | Three seed-0 arms: simulator only, paper input only, simulator + paper input | `dc98850` | Paper inputs move E3 price levels toward the paper; simulator-only changes little. No arm recovers the paper’s large recurrent–simpler gap. | **Completed sensitivity analysis; not a replacement formal baseline** |
| E4 Black–Scholes, Figure 10 | Five epsilon values, seeds 0–4, entropic lambda=1, 100k common OOS paths | seed 0: `feb6c7b`; seeds 1–4: `de1ddfc` | Per-seed log-log slope = 0.7129 ± 0.0017; all fits have R² about .999. | **Frozen — strong cross-seed approximate scaling** |
| E4 Heston, Figure 11 | Five epsilon values, seeds 0–4, entropic lambda=1, 100k common OOS paths | seed 0: `feb6c7b`; seeds 1–4: `de1ddfc` | Per-seed slope = 0.6469 ± 0.0173; median .6545. Seed 3 is lower at .6160 but passes engineering checks. | **Frozen — approximate two-thirds-type scaling with higher seed sensitivity** |

All “±” values above are cross-seed sample standard deviations.

## 3. E2 core seed registry

| Seed | OOS paths | Evaluation seed | Price alpha=.50 | Price alpha=.99 | Paper-target check |
|---:|---:|---:|---:|---:|---|
| 0 | 1,000,000 | 10001 | 1.943641 | 2.660551 | alpha=.50 pass; alpha=.99 fail |
| 1 | 1,000,000 | 10002 | 1.943900 | 2.655010 | alpha=.50 pass; alpha=.99 fail |
| 2 | 1,000,000 | 10003 | 1.942117 | 2.653323 | alpha=.50 pass; alpha=.99 fail |
| 3 | 1,000,000 | 10004 | 1.942959 | 2.666111 | alpha=.50 pass; alpha=.99 fail |
| 4 | 1,000,000 | 10005 | 1.943382 | 2.662276 | alpha=.50 pass; alpha=.99 fail |

The saved `passed: false` field for E2 is expected: all five runs fail the alpha=.99 absolute-price band while retaining the intended risk-ordering checks. Freeze this as a quantitative discrepancy, not as a failed execution.

## 4. Primary artifact index

| Experiment | Archived artifacts | Consolidated interpretation |
|---|---|---|
| E2 | `backups/e2/<seed>/` | this registry and `docs/method_replication-A2.md` |
| E2 call spread | `backups/e2cs_simple_seed0/`, `backups/e2cs_seeds1_4/` | `docs/e2cs_multiseed_summary.md` and `reports/e2cs/e2cs_multiseed_summary.json` |
| E3 | `backups/e3/<seed>/` | `docs/e3_multiseed_summary.md` and `reports/e3/e3_multiseed_summary.json` |
| Fidelity controls | `backups/e2e3_fidelity/` | `docs/e2e3_fidelity_controls.md` |
| E4 | `backups/e4bs/`, `backups/e4h/` | `docs/e4_robustness_analysis.md` and `reports/e4/e4_robustness_summary.json` |

Canonical report figures are stored under `figures/`. The corresponding plotting scripts read the saved results and do not retrain models.

## 5. Provenance notes

- E2 seeds 0–4 are clean runs from `85bdcbc`.
- E2cs seed 0 is a clean run from `1e75b7d`; seeds 1–4 are clean runs from `bfd7203`.
- E3 seed 0 is a clean run from `feb6c7b`. Seeds 1–4 record `f65e886` with `dirty: true`; their resolved config hashes are identical and the training implementation is unchanged, so they remain comparable. The dirty flag is retained as a limitation.
- E4 seed 0 is from `feb6c7b`; seeds 1–4 are clean runs from `de1ddfc`.
- The seed-0 fidelity controls are clean runs from `dc98850`.

## 6. Repository-size policy

Large pathwise `pnl.npz` arrays are intentionally excluded from the Git repository because they materially increase clone size without being required for the frozen summary tables. The repository retains the compact metrics, resolved configurations, logs, reports and canonical figures needed to audit the conclusions. Any full pathwise archives kept outside Git are supplementary, not required by this registry.

## 7. Final decision

E2–E4 are formally frozen. No additional GPU run is required before merging the A2 integration branch. Remaining work is documentation, pull-request review, merge-commit integration into this repository’s `main`, and later transplantation of the reviewed A2 files into a fresh clone/fork of the A1 group repository for the upstream pull request.
