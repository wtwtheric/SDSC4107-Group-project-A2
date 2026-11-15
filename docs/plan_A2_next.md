# A2 Next-Stage Implementation Plan

> **Status:** Working plan for the remaining A2 experiments  
> **Last updated:** 2026-11-15  
> **Scope:** E2 completion, E2 call spread, E3, E4-BS, E4-Heston, and the low-friction E4 extension.  
> **This document does not replace** `docs/plan.md`, `docs/spec.md`, or `docs/method_replication-A2.md`. It translates their experiment targets and the current A2 implementation into an actionable development sequence.

---

## 1. Purpose and relationship to existing documents

This file is the implementation-level plan for the remaining A2 work.

| Document | Authority |
|---|---|
| `docs/plan.md` | Research scope, paper targets, numerical acceptance criteria, and project-level decisions |
| `docs/spec.md` | Repository conventions, experiment IDs, result schemas, directory layout, and Git workflow |
| `docs/method_replication-A2.md` | A2 methodology, frozen QE-M simulator, variance-swap information set, fractional CVaR, and documented deviations |
| `docs/plan_A2_next.md` | Implementation sequence, code changes, tests, experiment gates, outputs, and operational rules for the remaining A2 work |

If a numerical target or scope decision appears inconsistent, `docs/plan.md` remains authoritative. If a file or directory convention appears inconsistent, follow `docs/spec.md` unless the team explicitly approves a change.

**A2 scope remains partial replication of Sections 5.1–5.3, with the planned Section 5.5 extension handled separately by the Hong Kong workstream. Section 5.4 (high-dimensional Heston, d=10) is not part of our replication.** We should not describe A2 as a broad simplification of the paper: the team has implemented several supporting modules and numerical checks of its own.

## 2. Current status snapshot

### 2.1 A1 foundation available to A2

The current `main` branch already contains the main reusable components:

- `src/replication/env_heston.py`: Heston d=1/d=2 market world, stock and variance-swap marks, features, payoff, transaction-cost accounting, and terminal loss.
- `src/replication/env_bs.py`: Black–Scholes d=1 world for E4-BS.
- `src/replication/net_recurrent.py`: semi-recurrent and simpler policies.
- `src/replication/losses.py`: CVaR and entropic-risk wrappers.
- `src/common/metrics.py`: shared fractional empirical CVaR, proportional transaction costs, indifference-price helper, and entropic risk.
- `src/replication/pricing.py`: Black–Scholes pricing and Heston characteristic-function pricing.
- `src/replication/e1.py`: reusable Heston market/config construction, policy construction, CVaR training loop, evaluation, and E1 runner.
- `src/replication/e2.py`: E2 risk-preference comparison, paired training seeds, common held-out paths for the two policies, cross-evaluation metrics, and archived positions.

A2 should reuse these components rather than duplicate the Heston simulator, network, or metrics implementation.

### 2.2 E2 core — currently running

As of 2026-11-15, the formal E2 run has progressed to its third seed. The training logs observed so far do not show a large departure from seed 0.

This is encouraging as a training-stability observation, but it is **not yet a numerical validation result**. Final judgement requires all five seeds and their formal out-of-sample evaluations.

| Item | Status |
|---|---|
| E2 objective | Heston d=2, zero transaction cost, α=0.50 vs α=0.99 |
| Simulator | Andersen QE-M, shared with E1 and planned E3 |
| Training recipe | A1 validated recipe; batch 1024, 30,000 iterations, cosine LR |
| CVaR estimator | Fractional empirical CVaR |
| Seeds | Formal run has reached seed 2; full batch not yet signed off |
| OOS evaluation | 1M paths per formal seed, as specified by the current A2 run plan |
| Validation | Pending full metrics and cross-seed summary |

Do not edit the running AutoDL workspace, overwrite its config, or change the source commit while the formal process is active. Record the exact Git commit SHA and the actual runtime config used by the runner when archiving the results.

The older seed-0 result recorded in Section 6 of `docs/method_replication-A2.md` belongs to the earlier ceil-CVaR run. It must not be presented as the final fractional-CVaR E2 result. Update that method note after the current formal E2 run has completed and been reviewed.

### 2.3 Frozen common settings for E1–E3

Unless a controlled diagnostic gives a clear reason to change them, retain:

| Setting | Frozen value |
|---|---:|
| Heston simulator | `pfhedge.stochastic.generate_heston`, Andersen QE-M |
| Rebalancing | 30 daily intervals, T = 30/365 |
| S₀ | 100 |
| V₀ = θ | 0.04 |
| κ | 1 |
| Vol-of-vol σ | 2 |
| ρ | −0.7 |
| Policy | Semi-recurrent, two hidden layers |
| Hidden width | 17 for d=2 |
| Normalisation | LayerNorm |
| Output scaling | `[1.0, 100.0]` for d=2 |
| Optimiser | Adam |
| Initial learning rate | 0.005 |
| LR schedule | Cosine, ending at 0.0005 |
| Training batch | 1024 |
| Training iterations | 30,000 |
| Formal seeds | 0–4 |
| CVaR estimator | Fractional boundary weighting |

E1–E3 must continue to use QE-M. Changing the simulator between experiments would confound the effects of risk preference, payoff design, or transaction costs with a change in path-generation methodology. QE-M is our documented common simulation scheme; it is not a claim that we have exactly reproduced the paper's CIR/Broadie–Kaya path construction.

For formal E2, E2 call spread, and E3 tail-risk reporting, use 1,000,000 out-of-sample paths, evaluated in chunks (default 25,000) to control memory. Development and smoke tests may use smaller samples. Keep the comparison policies on the same generated paths. Record the evaluation seed and path count in the run metadata.

---

## 3. Gate A — Finish and validate E2 core

### 3.1 Research question

Does changing the CVaR confidence level from α=0.50 to α=0.99 change the learned hedge in the way reported in Section 5.2, Figures 6–7?

This is a distributional trade-off, not a requirement that the α=0.99 policy performs better on every risk metric.

### 3.2 Frozen experiment

- Market: Heston d=2.
- Instruments: stock + paper-style variance swap.
- Payoff: European call, K=100.
- Transaction cost: ε=0.
- Policies: recurrent α=0.50 and recurrent α=0.99.
- Training: five seeds, 0–4.
- Evaluation: 1M common QE-M paths per seed.

The current `src/replication/e2.py` already trains both policies with paired training seeds and evaluates them on one common held-out path set. Keep this logic unless a reviewed numerical issue requires a change.

### 3.3 Numerical checks

Use the acceptance criteria in `docs/plan.md`:

1. α=0.99 indifference price: paper reference 3.49; target range **[3.2, 3.8]**.
2. α=0.50 reference price: target range **[1.80, 2.10]**.
3. The α=0.99-trained policy should have a higher realised CVaR50 than the α=0.50-trained policy.
4. The α=0.99-trained policy should have a lower realised CVaR99 than the α=0.50-trained policy.
5. The two strategies must be evaluated on the same OOS paths within each seed.
6. Report each seed, mean, standard deviation, and range. Do not rely on a single seed or on the similarity of training curves alone.

If the distributional crossing is absent, investigate the loss sign, cash translation, CVaR tail convention, simulator, and evaluation-path pairing before changing hyperparameters. If the crossing is present but the α=0.99 price is outside the paper range, record that as a quantitative discrepancy; do not tune the experiment solely to force the price to 3.49.

### 3.4 Required outputs

Per seed:

```text
results/e2/<seed>/
    metrics.json
    config.yaml
    run.log
    pnl.npz
    policy_alpha50.pt
    policy_alpha99.pt
```

The checkpoint files are experiment artifacts and should remain in the private backup/archive rather than being committed to Git if they are large or covered by `.gitignore`.

Final figures:

```text
figures/fig06_e2_own_price.png
figures/fig07_e2_common_q.png
```

Figure 6 should show the own-indifference-price hedging-error distributions. Figure 7 should show the common risk-neutral-price comparison. Captions must state which loss/hedging-error sign convention is plotted.

After completion:
- append the formal E2 row to `docs/experiments.md`;
- update the E2 status and final numerical table in `docs/method_replication-A2.md`;
- archive the run logs, configs, metrics, and `pnl.npz`;
- do not mark E2 as passed until all numerical checks have been reviewed.

---

## 4. Gate B — E2 call-spread experiment

### 4.1 Research question

Does the risk preference change the hedge-position shape for a bounded call-spread payoff, and does the α=0.99 policy exhibit the narrower position region described in the paper?

### 4.2 Contract and frozen market

The payoff is:

\[
Z=(S_T-100)^+-(S_T-101)^+.
\]

It is bounded between 0 and 1. Use:

| Setting | Value |
|---|---|
| Market | Same Heston d=2 market as E1/E2 |
| Simulator | Andersen QE-M |
| Instruments | Stock + variance swap |
| Transaction cost | ε=0 |
| Strikes | K₁=100, K₂=101 |
| Risk levels | α=0.95 and α=0.99 |
| Seeds | 0–4 |
| OOS evaluation | 1M common paths |

Do not add a new Heston simulator or create a duplicate Heston world solely for the call spread.

### 4.3 Implementation plan

Current `HestonWorld.payoff()` is hard-coded to a single European call. Add a small, backward-compatible payoff specification rather than copying the entire world class.

Suggested interface:

```python
HestonMarket(
    ...,
    payoff_type="call",
    strike2=None,
)
```

Supported payoff types:

- `"call"`: `(S_T - K1)^+` — preserve the current default and all E1/E2 behaviour.
- `"call_spread"`: `(S_T - K1)^+ - (S_T - K2)^+` — require `K2 > K1`.

Reject unsupported payoff names and invalid strike ordering with clear errors. Keep the payoff calculation in one place, and keep the default call payoff numerically unchanged.

Create a separate runner, `src/replication/e2cs.py`. Reuse `market_from_cfg`, `train_policy`, `policy_from_cfg`, `HestonWorld`, `SemiRecurrentPolicy`, and the shared fractional CVaR implementation where appropriate. Do not overload `e2.py` with call-spread-specific assumptions or E2-core validation checks.

Create:

```text
configs/exp/e2cs.yaml
tests/test_e2cs.py
```

Use the single `e2cs.yaml` as the shared base configuration and run it twice with α=0.95 and α=0.99 (through an explicit config override or CLI argument recorded in the resolved `config.yaml`). This follows the existing repository config convention. The runner should train the two policies using paired seeds and the same training recipe. Evaluate them on common QE-M paths. Save the recurrent policy checkpoints and the positions required for the strategy-shape figures.

### 4.4 Required unit tests

At minimum, test the payoff directly:

| S_T | Expected payoff |
|---:|---:|
| 99.0 | 0.0 |
| 100.0 | 0.0 |
| 100.5 | 0.5 |
| 101.0 | 1.0 |
| 102.0 | 1.0 |

Also test:
- the default `"call"` payoff is unchanged;
- invalid `K2 <= K1` is rejected;
- the payoff has shape `(batch_size,)`;
- the call-spread payoff remains in `[0, K2-K1]` for non-negative terminal spot values;
- the zero-cost terminal-loss calculation uses the selected payoff.

### 4.5 Strategy-shape analysis

The primary output is the hedge strategy, not just the indifference price.

Save and compare:

\[
\delta^{stock}_k,\qquad \delta^{varswap}_k
\]

for α=0.95 and α=0.99.

Because the policy is semi-recurrent, its position at time k depends on the preceding policy positions as well as the current market features. Do not construct a plot by feeding an isolated time-k state into the network while silently setting the recurrent input to zero.

Preferred extraction:
1. Generate a fixed QE-M visualisation path set.
2. Run each complete path through the frozen policy.
3. At a pre-specified time (initially the midpoint, k=15), examine positions conditional on the same spot and variance-swap-state bins.
4. Plot conditional medians and dispersion bands, and show the number of observations in each bin.
5. Use the same paths, bins, and plotting limits for α=0.95 and α=0.99.

A separate visualisation sample (e.g. 100k–200k paths) is acceptable if storing all 1M pathwise positions is unnecessarily expensive. The formal 1M OOS evaluation remains the source for tail-risk metrics.

The exact quantitative definition of “position band width” should be finalised against the paper's Figures 8–9 before it is used as an acceptance threshold. Do not invent a numerical threshold. Report the visual comparison and a clearly defined width statistic.

### 4.6 Outputs and acceptance

```text
results/e2cs/0.95/<seed>/
results/e2cs/0.99/<seed>/
```

Each run should contain `metrics.json`, `config.yaml`, `run.log`, `pnl.npz`, and the relevant policy checkpoint.

Final figures:

```text
figures/fig08_e2cs_stock_position.png
figures/fig09_e2cs_varswap_position.png
```

The paper's qualitative target is a narrower hedge-position region at α=0.99. There is no separate paper price target specified here. Report prices and risk metrics, but do not claim success from price proximity alone.

---

## 5. Gate C — E3 transaction costs: recurrent vs simpler

### 5.1 Research question

With proportional transaction costs, does the recurrent policy outperform the simpler policy that does not receive the previous hedge position?

### 5.2 Frozen experiment

| Setting | Value |
|---|---|
| Market | Heston d=2 |
| Simulator | Andersen QE-M |
| Instruments | Stock + variance swap |
| Payoff | European call, K=100 |
| Transaction cost | ε=0.01 |
| Risk level | α=0.99 |
| Policies | Recurrent vs simpler |
| Seeds | 0–4 |
| Training | Same budget and optimiser recipe for both policies |
| OOS evaluation | 1M common paths |

Use the cost formula in the existing shared implementation:

\[
c_k=\sum_i \epsilon_i
\left|\delta_{i,k}-\delta_{i,k-1}\right|S_{i,k},
\qquad \delta_{i,-1}=0.
\]

The first trade is charged; no forced liquidation is added at maturity. Costs are based on notional instrument value, not merely on the absolute change in the number of units.

### 5.3 Implementation plan

The following components already exist:
- `HestonWorld.cost()` calls the shared proportional-cost implementation.
- `SemiRecurrentPolicy` supports `recurrent=True` and `recurrent=False`.
- `train_policy()` in `e1.py` already optimises the configured CVaR objective.

Create a dedicated `src/replication/e3.py` runner and `configs/exp/e3.yaml`. Reuse the existing training and evaluation helpers, but do not simply register `e1.run` as E3: its E1-specific checks (including the zero-cost price band and zero-cost architecture comparison) are not valid E3 acceptance checks.

For a controlled comparison:
- reset the training RNG to the same seed before training each architecture, so both start from paired random streams;
- keep the training budget, batch size, market parameters, and objective identical;
- evaluate both policies on the same held-out paths;
- save both checkpoints and the pathwise losses/positions needed for analysis.

Use a fixed OOS evaluation seed across the five E3 model seeds if the runner supports it cleanly. This makes the cross-seed comparison less sensitive to differences in evaluation-path samples. Record the chosen evaluation seed explicitly. Do not change the active E2 run to achieve this.

### 5.4 Metrics and paper target

Produce the paper-style three-row comparison:

1. Mean terminal loss.
2. Indifference price.
3. Realised CVaR at α=0.99.

Also record:
- turnover;
- total transaction cost;
- transaction cost by instrument (stock leg and variance-swap leg);
- cross-seed mean, standard deviation, and range.

The instrument-level cost breakdown matters because the stock and variance-swap marks have very different price scales. Under the notional-cost convention, do not attribute the E3 effect to the variance-swap leg without reporting the measured cost contribution of each leg.

Paper reference prices from `docs/plan.md`:

| Policy | Reference indifference price | Target range |
|---|---:|---:|
| Recurrent | ≈5.51 | [5.2, 5.8] |
| Simpler | ≈6.74 | [6.4, 7.1] |

The expected qualitative result is that the recurrent policy has a lower indifference price and is strictly better under the paper's comparison. Preserve unrounded metric values, especially when values are close to zero. Do not replace small negative values with `-0.0` or round them away in the raw result files.

### 5.5 Outputs

```text
results/e3/<seed>/
    metrics.json
    config.yaml
    run.log
    pnl.npz
    policy_recurrent.pt
    policy_simpler.pt
```

Final comparison table/figure:

```text
figures/fig05_e3_architecture_comparison.png
```

Update `docs/experiments.md` and the A2 method note after the five-seed results have been reviewed.

---

## 6. Gate D — E4 entropic-risk asymptotics

### 6.1 Research question

Does the increase in indifference price caused by small proportional transaction costs scale approximately as \(\epsilon^{2/3}\), as in Section 5.3?

Use the entropic risk measure:

\[
\rho_\lambda(L)=\frac{1}{\lambda}
\log \mathbb{E}\left[e^{\lambda L}\right],
\qquad \lambda=1.
\]

Use the paper's five-point grid:

\[
\epsilon_i=2^{-(i+5)},\quad i=1,\ldots,5,
\]

i.e. \(2^{-6},2^{-7},2^{-8},2^{-9},2^{-10}\), approximately 156.3, 78.1, 39.1, 19.5, and 9.8 basis points.

### 6.2 E4-BS first

Market:
- Black–Scholes d=1.
- S₀=K=100.
- σ=0.2.
- T=30/365.
- One stock hedging instrument.
- Entropic λ=1.
- Five ε values and seeds 0–4.

Reuse `src/replication/env_bs.py`, `src/replication/losses.py`, and the analytic Black–Scholes price in `src/replication/pricing.py`.

### 6.3 E4-Heston second

Market:
- Heston d=2 with the frozen E1–E3 parameters.
- Stock + variance swap.
- QE-M simulator.
- Same maturity, rebalancing schedule, ε grid, and entropic λ=1.

Reuse `HestonWorld` and the Heston characteristic-function pricing implementation for the zero-cost reference price p₀. Do not replace the pricing benchmark with an unrelated approximation without documenting and validating the change.

### 6.4 Required implementation

The current repository already has:
- `BSWorld` and `HestonWorld`;
- `metrics.entropic()` and `entropic_loss()`;
- analytic BS pricing and Heston characteristic-function pricing.

However, the existing E1 `train_policy()` hard-codes the CVaR objective. E4 therefore needs a dedicated entropic training path or a carefully tested objective parameter in the shared trainer. Preserve the existing E1–E3 CVaR behaviour exactly.

Suggested files:

```text
src/replication/e4.py
configs/exp/e4bs.yaml
configs/exp/e4h.yaml
tests/test_e4.py
```

For each ε and seed:
1. Train a policy by minimising the entropic risk of terminal loss.
2. Evaluate on independent out-of-sample paths.
3. Compute pε using the same entropic-risk and cash-translation convention as the training objective.
4. Use the analytic/model price as p₀: Black–Scholes analytic price for E4-BS and Heston characteristic-function price for E4-Heston.
5. Save raw pε, p₀, pε−p₀, evaluation path count, and all seed-level results.

Use at least 100,000 OOS paths per run, as specified in `docs/plan.md`. If Heston estimates are too noisy, increase the evaluation count (the plan allows a larger sample, initially 200,000) before changing the training recipe. Do not automatically impose the 1M E2/E3 evaluation budget on E4 without a runtime/noise justification.

### 6.5 Regression and acceptance

For each model:
1. Aggregate pε across seeds at each ε.
2. Fit ordinary least squares to:

\[
\log(p_\epsilon-p_0)
=
a+\beta\log(\epsilon).
\]

3. Report β, R², the five mean price differences, and cross-seed standard deviations.

The paper targets in `docs/plan.md` are:
- E4-BS: β = 0.67 ± 0.04.
- E4-Heston: β = 0.71 ± 0.05.
- R² > 0.95.

Only fit the log regression if all five mean price differences are positive and numerically distinguishable from evaluation/training noise. If one or more differences are non-positive or indistinguishable from zero, report the limitation rather than adding an arbitrary offset before taking logs.

### 6.6 Outputs

```text
results/e4bs/<eps>/<seed>/
results/e4h/<eps>/<seed>/
```

Final figures:

```text
figures/fig10_e4bs_asymptotic_slope.png
figures/fig11_e4h_asymptotic_slope.png
```

Keep the underlying seed-level prices and regression inputs in machine-readable files. A plotted fitted line alone is not sufficient evidence.

---

## 7. Gate E — Low-friction E4 extension

This is an extension after the main E4 grid is stable, not a prerequisite for E4-BS or E4-Heston.

Follow the latest research plan in `docs/plan.md`:

\[
\epsilon\in\{2^{-11},2^{-12},2^{-13}\}
\]

approximately 4.9, 2.4, and 1.2 basis points.

Start with seeds 0–2. Add seeds 3–4 only if the seed-level uncertainty prevents a meaningful distinction.

For each ε, check whether:

\[
\operatorname{SD}_{seed}(p_\epsilon-p_0)
<
\frac{\mathbb{E}_{seed}[p_\epsilon-p_0]}{5}.
\]

If this condition is not met, do not report a low-friction slope as reliable. State that the 1bp-region effect cannot be resolved at the current numerical precision. That is a valid result.

**Documentation mismatch to resolve:** `docs/spec.md` currently contains an older E4-low grid in its experiment-ID table, while `docs/plan.md` v1.4 specifies the extension grid above. The research grid in `docs/plan.md` is authoritative. Before creating the formal E4-low run directory, update the relevant spec entry through a reviewed documentation change so that the repository does not contain two conflicting definitions.

---

## 8. Shared testing and quality requirements

Before formal training for each new experiment:

### E2 call spread
- Payoff unit tests in `tests/test_e2cs.py`.
- Regression test that the default European-call payoff is unchanged.
- Shape, strike-order, and payoff-bound tests.
- Smoke run with a small batch and a short iteration count.

### E3
- Test that recurrent and simpler policies both produce `(B, n_steps, d)` positions.
- Test positive transaction costs and first-trade charging.
- Test that the cost decomposition by instrument sums to the total cost.
- Smoke run for both architectures.

### E4
- Test entropic risk against a hand-computable loss vector.
- Test cash translation: `rho(L + c) = rho(L) + c`.
- Test that the BS zero-cost reference price agrees with the analytic price.
- Test the Heston p₀ interface against the existing pricing checks.
- Test that the trainer actually uses the entropic objective, not CVaR.
- Smoke run for both BS and Heston.

For every code change:

```bash
ruff check .
ruff format --check .
pytest -q
```

Do not weaken or delete existing tests to make a new experiment pass. If a test assumption is environment-specific (for example CPU vs CUDA), fix the test to select the intended device explicitly rather than changing production behaviour.

---

## 9. Results, figures, and reproducibility

### 9.1 Per-run result structure

Follow `docs/spec.md`:

```text
results/<exp>/<...>/<seed>/
    metrics.json
    config.yaml
    run.log
    pnl.npz
    fig_*.png   # optional per-run diagnostic figures
```

For each run:
- `config.yaml` must be the actual resolved runtime configuration, not merely a copy of the source YAML.
- `metrics.json` must record seed, commit SHA, dirty state, device, runtime, path count, metrics, and checks.
- `run.log` must preserve the original stdout/stderr.
- `pnl.npz` must contain the pathwise losses required to reproduce the reported risk metrics.
- Save model checkpoints as private experiment artifacts; do not commit large weights to Git.

### 9.2 Final figure paths

The final report figures live in `figures/`, following the existing repository convention:

```text
figures/fig05_e3_architecture_comparison.png
figures/fig06_e2_own_price.png
figures/fig07_e2_common_q.png
figures/fig08_e2cs_stock_position.png
figures/fig09_e2cs_varswap_position.png
figures/fig10_e4bs_asymptotic_slope.png
figures/fig11_e4h_asymptotic_slope.png
```

Use PNG, 300 dpi, English axis labels and legends. Put the conclusion in the caption rather than embedding a large title in the plot. Include paper reference values where relevant. Do not make a figure from an unvalidated debug run.

### 9.3 Experiment registry and backups

After each formal experiment is reviewed:
- append one row to `docs/experiments.md`;
- include experiment ID, seed/ε/cost, exact commit, runtime, one-sentence result, and pass/partial/fail status;
- make every figure and table traceable to a registry row;
- back up results and checkpoints outside Git (private storage or an archived ZIP);
- keep large result files and private datasets out of the public repository.

Never describe a run as reproducible if its source commit, resolved config, seed, or evaluation path count is missing.

---

## 10. Execution order and gates

| Gate | Work | Exit condition |
|---|---|---|
| A | Finish E2 core seeds 0–4 | All five formal runs and 1M OOS metrics reviewed; distributional crossing and price checks reported |
| B | Implement E2 call spread | Payoff tests, smoke tests, and policy-position extraction validated |
| C | Run E2 call spread | α=0.95/0.99 results and Figures 8–9 reviewed across seeds |
| D | Implement and run E3 | Recurrent/simpler comparison, cost accounting, and five-seed summary complete |
| E | Implement and run E4-BS | Five ε levels, seed-level prices, and BS slope reviewed |
| F | Implement and run E4-Heston | Five ε levels, seed-level prices, and Heston slope reviewed |
| G | Low-friction extension | Precision criterion assessed; report slope only if resolvable |
| H | Final A2 synthesis | Method note, experiment registry, figures, and findings draft agree |

### Work that can proceed while E2 is running

Safe parallel preparation:
- draft E2 call-spread payoff tests and the separate runner design;
- prepare E3 config and test design;
- prepare the E4 entropic-objective tests and regression utility;
- review the original paper Figures 5 and 8–11 to finalise figure extraction and captions;
- draft the result-aggregation scripts without touching the active run directory.

Do **not**:
- change the active E2 source checkout or runtime config;
- overwrite files used by the AutoDL process;
- start formal E2CS/E3/E4 training before their code and smoke tests pass;
- change the QE-M simulator or the frozen E1–E3 training recipe merely because a later seed differs slightly;
- tune parameters only to move results closer to paper reference values.

---

## 11. Recommended development branches

Follow the branch and PR conventions in `docs/spec.md`. Keep the formal AutoDL run pinned to its recorded commit.

Suggested isolated branches:

```text
feat/a2-call-spread
feat/a2-e3
feat/a2-e4
```

Each branch should:
1. start from the reviewed A2 baseline commit;
2. contain one logically grouped experiment implementation;
3. pass Ruff and pytest;
4. include config and tests with the runner;
5. be reviewed before merging into the stable branch.

The plan document itself can be committed separately as a documentation-only change. Do not merge experimental code into the active formal-run checkout while its process is running.

---

## 12. Immediate next actions

1. Let the current E2 formal run finish without modifying its runtime environment.
2. Collect all five `run.log`, `metrics.json`, resolved `config.yaml`, and 1M evaluation outputs.
3. Confirm the exact source commit and whether all five seeds used the fractional-CVaR implementation.
4. Produce a seed-level table and cross-seed mean ± standard deviation for the E2 price and both realised CVaR measures.
5. Update `docs/method_replication-A2.md` and `docs/experiments.md` with the reviewed formal E2 results.
6. In a separate development branch, implement the call-spread payoff abstraction and its tests.
7. Run E2CS smoke tests; only then start formal E2CS seeds.
8. Continue to E3, E4-BS, E4-Heston, and the low-friction extension in the gate order above.

**Decision rule:** a stable-looking training curve is necessary but not sufficient. A formal experiment is complete only when its implementation, numerical checks, OOS evaluation, saved artifacts, and provenance all agree.
