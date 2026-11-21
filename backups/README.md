# Backups

This directory is the opt-in, version-controlled backup location for selected
experiment artifacts and datasets. It is separate from the working directories:

- `results/` remains gitignored and is the reproducible runtime output area.
- `data/` remains gitignored and is the local working-data area.
- `figures/` remains the canonical, tracked report-figure directory.
- `backups/e2/<seed>/` stores selected E2 checkpoints and lightweight run
  metadata needed to reproduce later evaluation.
- `backups/e2cs_simple_seed0/` stores the corrected E2cs seed-0 checkpoints and
  figures used by the detailed diagnostic.
- `backups/e2cs_seeds1_4/` stores the frozen-recipe E2cs seed 1–4 metrics,
  figures, logs, and independent BatchNorm recalibration reports. The
  consolidated five-seed interpretation is in `docs/e2cs_multiseed_summary.md`.
- `backups/e3/` stores the E3 baseline artifacts for seeds 0–4. Seed 0 also retains the independent-evaluation diagnostic under `independent_evaluation_diagonostics/`; seeds 1–4 include the integrated evaluation outputs and Figure 5. The consolidated interpretation is in `docs/e3_multiseed_summary.md`.
- `backups/e2e3_fidelity/` stores the three seed-0 simulator/input control arms used for the E2/E3 fidelity sensitivity analysis.
- `backups/e4bs/` and `backups/e4h/` store the compact five-seed E4 epsilon summaries used by the frozen robustness analysis.
- `docs/experiments-e2-e4.md` is the A2 experiment registry and provenance index; the A1-owned `docs/experiments.md` is intentionally separate.
- `backups/data/` stores only datasets intentionally selected for archival.

The repository is public. Files copied here become public. Use Git LFS for
large checkpoints and datasets. Large pathwise `pnl.npz` arrays are intentionally not tracked because they are unnecessary for the frozen summary and slow repository clones. Keep a source/licence note,
download date, and SHA-256 manifest with archived datasets. Do not commit
confidential or restricted data.
