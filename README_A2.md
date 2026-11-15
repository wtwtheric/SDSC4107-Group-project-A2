# A2 Experiment Plan and Progress

> Last updated: 2026-11-15  
> Scope: E2 / E2 Call Spread / E3 / E4 implementation, experiments, validation and next actions.  
> Scientific scope and acceptance criteria remain governed by `docs/plan.md`; collaboration and delivery rules remain governed by `docs/spec.md`.

## 1. A2 objective and execution order

1. **E2 core** — Heston d=2, zero costs; CVaR 50% vs 99% (Figures 6–7).
2. **E2 Call Spread** — normalized call spread; CVaR 95% / 99% policies and model hedge (Figures 8–9).
3. **E3** — Heston d=2, `eps=0.01`, `alpha=0.99`; recurrent vs simpler (Figure 5).
4. **E4-BS** — Black–Scholes d=1, entropic risk, epsilon sweep and log–log slope.
5. **E4-Heston** — repeat the asymptotic experiment in Heston d=2.
6. **Low-friction extension** — smaller epsilon values only after the main grid, and report a slope only if signal-to-noise is adequate.

E2 core is implemented and has seed-0 training plus 1M-path evaluation archived. Numerical validation is not complete; diagnose the discrepancy before treating E2 as finished.

## 2. A1 baseline inherited by A2

Unless a controlled diagnostic justifies a documented deviation, retain the repository's validated E1 engineering recipe for E2/E3.

| Setting | Baseline |
|---|---:|
| Heston S0=K | 100 |
| V0=theta | 0.04 |
| kappa / vol-of-vol sigma / rho | 1 / 2 / -0.7 |
| Maturity / steps | 30/365 / 30 daily steps |
| Instruments | Stock + paper-style variance swap |
| Policy | Semi-recurrent; independent network per trading date |
| Hidden width | 17, 17 |
| Normalization | LayerNorm (documented deviation from paper BatchNorm) |
| Output scale | [1, 100] |
| Optimizer | Adam |
| Learning rate | 0.005 -> cosine -> 0.0005 |
| Batch / iterations | 1024 / 30,000 |
| Standard evaluation paths | 100,000 |
| Device | `auto`; record actual device and PyTorch version |

These are repository-validated engineering settings, not a claim that the paper fully disclosed all training hyperparameters. Kaggle changes execution environment, not scientific config.

## 3. Current progress (2026-11-15)

### 3.1 GitHub status

- Repository: https://github.com/wtwtheric/SDSC4107-Group-project-A2
- E2 implementation and seed-0 1M evaluation are merged into `main`.
- PR #3: https://github.com/wtwtheric/SDSC4107-Group-project-A2/pull/3 — merged.
- Merge commit: `a459ca49cf995ec19431e066edbf526578aa7035`.
- Main CI run #34: https://github.com/wtwtheric/SDSC4107-Group-project-A2/actions/runs/43615805805 — success.
- CI success confirms automated code checks/tests, not numerical replication.

### 3.2 E2 files and artifacts merged

- `src/replication/e2.py` — trains CVaR 50% and 99% policies and evaluates both on common held-out paths.
- `configs/exp/e2.yaml` — E2 config.
- `tests/test_e2.py` — targeted runner/schema/common-path/price-translation tests.
- `scripts/evaluate_e2.py` — evaluates frozen checkpoints on fresh common Heston paths in batches.
- `scripts/make_e2_figures.py` — generates Figures 6 and 7.
- `docs/e2_kaggle.md` — Kaggle execution and GitHub backup instructions.
- `backups/e2/0/` — seed-0 checkpoints, config and training metrics.
- `backups/e2/0/evaluation_1000000/` — 1M evaluation metrics, PnL archive, config and summary CSV.
- `figures/fig06_e2_own_price.png` and `figures/fig07_e2_common_q.png` — committed figures.

### 3.3 E2 seed-0 1M-path evaluation

1,000,000 fresh paths; batch size 25,000; same generated path sample for both frozen policies.

| Metric | CVaR 50% policy | CVaR 99% policy |
|---|---:|---:|
| Indifference price | 1.942499 | 2.652585 |
| Realized CVaR 50% of (L-q) | 0.250665 | 0.488328 |
| Realized CVaR 99% of (L-q) | 1.892440 | 0.960751 |
| Mean absolute hedging error E|q-L| | 0.251724 | 0.527780 |
| Mean raw loss E[L] | 1.691559 | 1.691964 |

Risk-neutral price: q=1.691834.

Paper Section 5.2 reference:

- p_0.50=1.94; p_0.99=3.49.
- CVaR99-trained policy: realized CVaR50=0.527, CVaR99=1.8034.
- CVaR50-trained policy: realized CVaR50=0.2531, CVaR99=2.3631.

Interpretation:
- CVaR50 price is close to the paper reference.
- Cross-risk trade-off has the expected direction: CVaR99 policy has higher realized CVaR50 but lower realized CVaR99 than CVaR50 policy.
- However, CVaR99 policy price (2.6526) is materially below 3.49; its realized CVaR99 (0.9608) is also below 1.8034.
- E2 is **implemented and evaluated, but not yet numerically accepted as a full replication**.

The implementation reports `mean_abs_hedging_error = mean(|q-L|)`. This operational metric must not be presented as identical to the paper table's “Mean Loss” without verifying the original definition.

### 3.4 Automated evaluation checks

| Check | Status |
|---|---|
| CVaR99 price in [3.2, 3.8] | FAIL |
| CVaR50 reference price in [1.80, 2.10] | PASS |
| CVaR50 cross-risk trade-off | PASS |
| CVaR99 cross-risk trade-off | PASS |
| Price / cash-translation identities | PASS |

Do not weaken or remove the price check merely to obtain `passed=true`. Explain the discrepancy first.

### 3.5 Reproducibility caveat

Archived seed-0 training `metrics.json` records `git_commit: null` and `dirty: true`. The 1M evaluation records checkpoint SHA-256 hashes and its evaluation commit, but source training commit is also null. Checkpoints and outputs are preserved, but the exact clean source commit used for training is not fully established. New formal runs must start from a clean, recorded commit and preserve config/checkpoint hashes.

## 4. E2 discrepancy: diagnostic gate before Call Spread

The 100k and 1M CVaR99 prices are similar (about 2.66), so ordinary evaluation sample size alone is unlikely to explain the gap. Do not immediately increase iterations or tune parameters to hit 3.49.

Investigate one factor at a time:

1. **Risk measure and loss convention** — independently verify upper-tail CVaR, L=Z-hedge gain, alpha=0.99, and p_alpha=q+CVaR_alpha(L-q). Existing translation identity checks pass, but retain an independent numerical check.
2. **Heston path-generation deviation** — paper describes exact sampling from the CIR transition density followed by a simplified Broadie–Kaya scheme. Repository uses PFHedge `generate_heston` (Andersen QE-M). This documented difference may matter for tail-sensitive metrics.
3. **State representation** — paper describes inputs based on (log S_t, V_t), with variance-swap price as an alternative. Repository uses [log(S_t/S0), t/T, S2_t/S2_0]. Confirm and document implications.
4. **Network normalization** — paper uses BatchNorm; repository uses LayerNorm due to A1 stability diagnostics. Do not change without controlled comparison.
5. **CVaR99 minibatch noise** — batch 1024 contains about 10 observations in the worst 1% tail. Assess whether this creates noisy tail gradients. If justified, compare a larger batch as a separate diagnostic.
6. **Training convergence** — inspect training logs/checkpoints if available. The 30k recipe was validated for E1; that does not prove convergence for CVaR99.

Verify implementation and simulation conventions first. Then run the smallest controlled experiment that distinguishes hypotheses. Do not launch an expensive multi-seed sweep without a specific diagnostic reason.

## 5. Next experiment: E2 Call Spread (Figures 8–9)

### 5.1 Paper specification

The paper defines the normalized call spread:

Z = [(S_T-K1)^+ - (S_T-K2)^+] / (K2-K1), K1<K2,

with K1=S0=100 and K2=101. The denominator is 1 in this experiment, but retain it in implementation.

- Same Heston d=2 market and stock + variance-swap hedging instruments.
- Zero transaction costs.
- Compare model hedge (paper's alpha=0.5 reference) with deep-hedging policies trained at alpha=0.95 and alpha=0.99.
- Plot stock hedge component at t=15 days over an (S,V) grid, comparing neural policy with model call-spread delta.
- Paper's qualitative finding: higher risk aversion flattens the strategy, interpreted as a barrier shift. Do not simplify this to an unsupported claim that every position range merely narrows.

### 5.2 Planned implementation

Reuse Heston simulator, network and training code. Add payoff specification to market/config rather than duplicating the entire Heston world.

Proposed config:
- Payoff type: `call_spread`.
- K1=100, K2=101.
- Normalize by K2-K1.
- alpha=0.95 and alpha=0.99.
- eps=0.0, Heston d=2.

Targeted tests:
1. Existing European-call payoff remains unchanged.
2. For S_T=99, 100.5, 102, normalized spread payoff is 0, 0.5, 1.
3. Invalid strikes (K2<=K1) raise a clear error.
4. Model call-spread delta equals difference of two call deltas divided by K2-K1, and agrees with finite-difference price check.
5. Stock-delta surfaces have finite values and consistent (S,V) grid dimensions.

Start with seed 0 and a short smoke run. Only after payoff, model-delta and surface checks pass should formal training begin. Keep simulator, feature and normalization deviations explicit.

### 5.3 Suggested code scope

- Extend `HestonMarket` / config parsing with backward-compatible payoff definition.
- Extend `HestonWorld.payoff()` to support call and normalized call spread.
- Extend benchmark calculation to produce model call-spread delta as difference of two call deltas.
- Add a separated Call Spread runner/config or a clearly isolated mode.
- Add Figure 8/9 plotting without mixing Figure 6/7 histogram outputs.
- Preserve existing European-call defaults for E1/E2.

## 6. Remaining A2 roadmap

| Stage | Main task | Acceptance focus | Status |
|---|---|---|---|
| E2 core | CVaR50 vs CVaR99 European call | Price target + cross-risk distribution | Implemented; numerical discrepancy open |
| E2 Call Spread | K1=100, K2=101; alpha=.95,.99 | Stock-delta surface and flattening/barrier-shift pattern | Planned |
| E3 | eps=.01, alpha=.99; recurrent vs simpler | Recurrent advantage and Figure 5 table | Not started |
| E4-BS | Entropic lambda=1; epsilon sweep | Slope around 0.67 | Not started |
| E4-Heston | Same sweep in Heston d=2 | Slope around 0.71 | Not started |
| Low-friction extension | Add 2^-11, 2^-12, 2^-13 | Report slope only if signal-to-noise gate passes | Not started |

## 7. Coding, testing and experiment rules

- Reuse A1 modules; do not rewrite market simulation, network or training code without demonstrated need.
- Keep scientific settings in YAML. Kaggle Notebook clones, installs/verifies dependencies and invokes repository commands; it should not duplicate experiment settings.
- Use risk-based verification: targeted tests for changed behavior; full CI when a shared interface or cross-module behavior changes.
- Do not run long training, multi-seed sweeps or large data generation as default code checks. Obtain confirmation before expensive GPU runs and state purpose/cost.
- Never weaken assertions or acceptance criteria just to make a run pass.
- Record each formal run's clean git commit, config hash, seed, simulator, device, torch version, iterations, batch size, train/test path counts, checkpoint hashes and output paths.
- Keep generated run files under `results/`; commit selected reproducibility artifacts under `backups/`; canonical report figures under `figures/`.
- The GitHub repository is public. Never upload secrets or restricted data.

## 8. Environment management note

The repository currently uses `requirements.lock.txt` as its practical dependency specification. `pyproject.toml` contains tool configuration but no complete project dependency declaration; current `uv.lock` is only a minimal stub, not a complete dependency lock. Until deliberate uv migration is completed, do not rely on `uv sync` to recreate the project environment. A temporary uv-managed environment can be created with `uv venv` and populated from the existing requirements file, while respecting platform-specific PyTorch/CUDA installation.

## 9. Immediate next actions

1. Pull merged `main` locally and confirm working tree is clean.
2. Preserve/recover E2 seed-0 training log if available and keep exact checkpoints.
3. Complete E2 CVaR99 discrepancy diagnostics in Section 4; do not launch a costly sweep yet.
4. Once simulator/risk-measure/training explanation is understood, implement Call Spread payoff and model-delta support on a new feature branch.
5. Run targeted tests and a seed-0 smoke experiment before formal GPU training.
