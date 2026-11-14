# E2 Kaggle workflow and artifact backup

This workflow reuses the trained E2 seed-0 policies. It does **not** retrain the
30,000-iteration networks. Canonical plots are written to the repository-root
`figures/` directory; machine-generated run files remain under `results/`.

## 1. Prepare the E2 checkpoint backup locally

The working `results/` directory is gitignored. Copy the files needed to
resume evaluation into the tracked backup area:

```bash
mkdir -p backups/e2/0
cp results/e2/0/config.yaml backups/e2/0/
cp results/e2/0/metrics.json backups/e2/0/
cp results/e2/0/policy_alpha50.pt backups/e2/0/
cp results/e2/0/policy_alpha99.pt backups/e2/0/
```

The two `.pt` files are required to evaluate without retraining. The 100k
`pnl.npz` is optional because the 1M evaluation generates a fresh test set.

This repository is public. Anything committed here becomes publicly
downloadable. Do not upload licensed, confidential, personal, or restricted
data. GitHub blocks ordinary Git files larger than 100 MB; use Git LFS for
checkpoints, large `.npz` files, and datasets:

```bash
git lfs install
git lfs track "backups/e2/**/*.pt" "backups/e2/**/*.npz" "backups/data/**"
git add .gitattributes backups/e2/0
git commit -m "data(e2): archive seed 0 checkpoints"
git push origin exp/e2-output-eval
```

For dataset backups, copy selected files into `backups/data/` rather than
force-adding the ignored working `data/` directory. Keep source, download
date, licence/terms, and SHA-256 checksums alongside archived data.

## 2. Clone the repository in Kaggle

Turn **Internet** on in Kaggle Notebook settings. The public repository can be
cloned without a token:

```bash
!git clone --branch exp/e2-output-eval --single-branch https://github.com/wtwtheric/SDSC4107-Group-project-A2.git
%cd /kaggle/working/SDSC4107-Group-project-A2
```

If the checkpoints were stored with Git LFS:

```bash
!apt-get -qq update && apt-get -qq install -y git-lfs
!git lfs install
!git lfs pull
```

Install dependencies without replacing Kaggle's preinstalled CUDA PyTorch
build. Do not blindly install the full lock file, which pins the project's
local PyTorch build:

```bash
!pip install "numpy==1.26.4" "scipy==1.13.1" "pfhedge==0.23.0" "matplotlib==3.11.2" "PyYAML==6.0.3"
```

Check the runtime:

```python
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## 3. Run the 1M-path evaluation and render the figures

The evaluator loads frozen checkpoints from `backups/e2/0/`, simulates fresh
common Heston paths in chunks, and stores terminal losses plus per-step
position summaries. It does not retain all 1M pathwise positions or retrain.

```bash
!python scripts/evaluate_e2.py \
  --run-dir backups/e2/0 \
  --output-dir results/e2/0/evaluation_1000000 \
  --n-paths 1000000 \
  --batch-size 25000 \
  --device auto
```

Then render Figure 6 and Figure 7:

```bash
!python scripts/make_e2_figures.py \
  --evaluation-dir results/e2/0/evaluation_1000000
```

Expected outputs:

```text
results/e2/0/evaluation_1000000/
  config.yaml
  metrics.json
  pnl.npz
  fig07_e2_summary.csv

figures/
  fig06_e2_own_price.png
  fig07_e2_common_q.png
```

Figure 6 plots each strategy's hedging error at its own indifference premium,
`p_alpha - L`. Figure 7 charges the common risk-neutral premium and plots
`q - L`. CVaR is calculated on the upper tail of the loss `L - q`, not on
the sign-reversed plotted hedging error.

The paper labels the table column "Mean Loss" but does not state a separate
formula beside that table. This implementation uses the transparent
operational definition `mean(abs(q - L))`, labelled
`mean_abs_hedging_error`; realized CVaRs are computed from `L - q`.

## 4. Push selected results back to GitHub from Kaggle

Create a fine-grained GitHub token with **Contents: Read and write** permission
for this repository. Add it in Kaggle as a Notebook Secret named
`GITHUB_TOKEN`. Never paste it into a notebook cell or commit it.

Copy lightweight evaluation outputs into the tracked backup area. The full
1M `pnl.npz` is optional but can be archived using Git LFS:

```bash
mkdir -p backups/e2/0/evaluation_1000000
cp results/e2/0/evaluation_1000000/metrics.json backups/e2/0/evaluation_1000000/
cp results/e2/0/evaluation_1000000/config.yaml backups/e2/0/evaluation_1000000/
cp results/e2/0/evaluation_1000000/fig07_e2_summary.csv backups/e2/0/evaluation_1000000/
cp results/e2/0/evaluation_1000000/pnl.npz backups/e2/0/evaluation_1000000/
cp figures/fig06_e2_own_price.png figures/fig07_e2_common_q.png backups/e2/0/
```

Push on a result branch, not directly to `main`:

```bash
git checkout -b exp/e2-seed0-1m-results
git add figures/ backups/e2/0/
git commit -m "exp(e2): add 1m evaluation and figures"
git push -u origin exp/e2-seed0-1m-results
```

In a Kaggle Python cell, retrieve the token from Kaggle Secrets and pass the
authorization header only to the `git push` subprocess. This avoids writing
the token into `.git/config`:

```python
import base64
import os
import subprocess
from kaggle_secrets import UserSecretsClient

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
env = os.environ.copy()
env["GIT_CONFIG_COUNT"] = "1"
env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {auth}"

subprocess.run(
    ["git", "push", "-u", "origin", "exp/e2-seed0-1m-results"],
    cwd="/kaggle/working/SDSC4107-Group-project-A2",
    env=env,
    check=True,
)
```

Open a pull request from the result branch into `main`. Inspect figures and
provenance, and merge only after CI passes.
