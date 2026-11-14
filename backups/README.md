# Backups

This directory is the opt-in, version-controlled backup location for selected
experiment artifacts and datasets. It is separate from the working directories:

- `results/` remains gitignored and is the reproducible runtime output area.
- `data/` remains gitignored and is the local working-data area.
- `figures/` remains the canonical, tracked report-figure directory.
- `backups/e2/<seed>/` stores selected E2 checkpoints and lightweight run
  metadata needed to reproduce later evaluation.
- `backups/data/` stores only datasets intentionally selected for archival.

The repository is public. Files copied here become public. Use Git LFS for
large checkpoints, `.npz` arrays, and datasets. Keep a source/licence note,
download date, and SHA-256 manifest with archived datasets. Do not commit
confidential or restricted data.
