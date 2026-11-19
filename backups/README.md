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
- `backups/data/` stores only datasets intentionally selected for archival.

The repository is public. Files copied here become public. Use Git LFS for
large checkpoints, `.npz` arrays, and datasets. Keep a source/licence note,
download date, and SHA-256 manifest with archived datasets. Do not commit
confidential or restricted data.
