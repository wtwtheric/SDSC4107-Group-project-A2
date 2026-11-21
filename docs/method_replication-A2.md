# Method note — A2 replication (E2–E4)

> Scope: A2 only.  
> The original `docs/method_replication.md` remains the A1 method record and is intentionally left unchanged.  
> This note records A2 methodology, frozen implementation choices, controlled diagnostics and experiment status for E2–E4.

## 1. Replication scope

The group project reproduces the four core Section 5 results defined in `docs/plan.md`:

- E1: zero-cost Heston baseline and model hedge comparison — A1.
- E2: risk-preference comparison under zero transaction costs, including the call-spread experiment — A2.
- E3: recurrent versus simpler architecture under proportional transaction costs — A2.
- E4: the small-cost pricing asymptotic in Black–Scholes and Heston — A2.

Section 5.4's high-dimensional (d=10) experiment is deliberately excluded. The real-market Section 5.5 idea is extended to Hong Kong data by the B group rather than reproduced literally.

## 2. Heston simulator frozen for E1–E3

For consistency across the replication chain, E1, E2 and E3 use the same Heston simulator:

- `pfhedge.stochastic.generate_heston`;
- Andersen quadratic-exponential scheme with martingale correction (QE-M);
- 30 daily rebalancing intervals over (T=30/365);
- (S_0=100), (V_0=\theta=0.04), (\kappa=1), vol-of-vol (=2), (\rho=-0.7).

This is a documented deviation from the paper. Section 5.2.2 describes exact sampling of the CIR transition density followed by a simplified Broadie–Kaya construction for the stock. We do **not** change the simulator between E1 and E2/E3 because doing so would confound the effect of risk preference or transaction costs with a change in path-generation methodology.

QE-M should therefore be read as the common simulation environment of our partial replication, not as a claim that it is the paper's exact discretisation.

## 3. Variance swap and network information set

The second hedging instrument is the variance-swap mark

[
S_t^2
=
\mathbb E_t\left[\int_0^T V_s\,ds\right]
=
\int_0^t V_s\,ds + L(t,V_t),
]

with

[
L(t,v)
=
\frac{v-\theta}{\kappa}\left(1-e^{-\kappa(T-t)}\right)+\theta(T-t).
]

It is the expected total integrated variance to maturity and is a martingale under the risk-neutral model. It should not be confused with an annualised variance rate.

The paper's displayed network input in Section 5.2.2 is

[
I_k=(\log S_k^1,V_k),
]

but the paper explicitly notes that (V_k) can instead be replaced by (S_k^2). Our A1–A3 Heston pipeline therefore uses

[
x_k =
\left[
\log(S_k/S_0),\;
t_k/T,\;
S_k^2/S_0^2
\right].
]

This choice is within the alternative information-set design explicitly mentioned by the paper, although our normalisation and explicit time feature are implementation choices.

(S_k^2) contains both accrued integrated variance and the conditional expectation of remaining variance. It therefore summarises some path history, but it is not simply "realised variance". Any claim that this feature improves tail risk relative to (V_k) requires a controlled feature ablation; the current E2 result alone does not establish that conclusion.

In `env_heston.py`, the accrued interval contribution is computed from the CIR conditional first moment rather than a left-rectangle approximation. This preserves the martingale property of the variance-swap mark in the discrete implementation and avoids a spurious drift that the optimiser could exploit.

## 4. Loss, CVaR and indifference price

For a liability (Z), hedging positions (\delta) and proportional costs (C_T(\delta)), the repository uses terminal loss

[
L
=
Z-(\delta\cdot S)_T+C_T(\delta).
]

The paper writes the corresponding terminal P&L with the opposite sign. Using the upper tail of (L) is therefore equivalent to applying the paper's lower-tail risk measure to P&L.

At zero cost the risk-neutral price is (q), and cash translation gives

[
p_\alpha
=
q+\mathrm{CVaR}_\alpha(L-q)
=
\mathrm{CVaR}_\alpha(L).
]

For E2, Figure 7 is plotted as the hedging error (q-L), while CVaR is computed on the upper tail of (L-q). The sign difference between the plotted hedging error and the risk-measure input is intentional.

## 5. Fractional empirical CVaR

The repository previously estimated empirical CVaR by taking

[
k=\left\lceil(1-\alpha)N\right\rceil
]

whole observations from the upper loss tail and averaging them. This is exact only when ((1-\alpha)N) is an integer.

A2 now uses a fractional boundary observation so that the empirical estimator matches the quantile-integral definition. Let

[
m=(1-\alpha)N,\qquad
k=\lfloor m\rfloor,\qquad
r=m-k,
]

and order losses from worst to best,

[
L_{(1)}\ge L_{(2)}\ge\cdots\ge L_{(N)}.
]

Then

[
\widehat{\mathrm{CVaR}}_\alpha
=
\frac{
\sum_{i=1}^{k}L_{(i)}+rL_{(k+1)}
}{m}.
]

If (r=0), this reduces to the ordinary mean of the worst (k) observations.

For the E2 training batch (N=1024):

- (\alpha=0.50): (m=512), so the old and new estimators are identical.
- (\alpha=0.99): (m=10.24), so the new estimator uses the worst 10 observations at full weight plus 0.24 of the 11th.
- (\alpha=0.95): (m=51.2), relevant to the call-spread experiment.

The old ceil implementation at (\alpha=0.99) averaged 11 complete observations, corresponding to a slightly wider empirical tail. Fractional weighting places relatively more weight on the most extreme ten observations and less weight on the boundary observation.

Synthetic numerical diagnostics show that the point-estimate difference is small relative to minibatch-to-minibatch CVaR noise, but gradient weights are different. The estimator change therefore needs a controlled retraining experiment rather than a post-hoc adjustment of the old checkpoint.

Tests cover:

1. integer tail mass;
2. non-integer tail mass;
3. autograd weights on the full and fractional boundary observations;
4. cash/translation invariance.

## 6. E2 seed-0 baseline before the estimator change

The original seed-0 E2 run used the ceil-boundary training estimator with batch 1024 and 30,000 iterations.

The archived 1M-path evaluation produced approximately:

| Metric | CVaR50 policy | CVaR99 policy |
|---|---:|---:|
| Indifference price | 1.9425 | 2.6526 |
| Realized CVaR50 of (L-q) | 0.2507 | 0.4883 |
| Realized CVaR99 of (L-q) | 1.8924 | 0.9608 |

The qualitative risk-preference crossing is present, but (p_{0.99}=2.6526) is below the paper reference 3.49. This remains an open quantitative discrepancy.

The seed-0 training log shows that the (\alpha=0.50) objective is essentially flat by the end. The (\alpha=0.99) moving average continues to decline slowly through 30,000 iterations, so it is best described as near-converged rather than proven fully converged.

## 7. Controlled Fractional-CVaR experiment

The estimator comparison changes **only** the empirical CVaR boundary treatment.

Frozen factors:

- simulator: Andersen QE-M;
- market parameters;
- seed: 0;
- network architecture and LayerNorm choice;
- feature map;
- output scaling;
- batch: 1024;
- optimiser and cosine learning-rate schedule;
- 30,000 training iterations.

Because (\alpha=0.50) has integer tail mass at batch 1024, it is not retrained for this diagnostic. Only the (\alpha=0.99) policy is retrained.

Run:

```bash
python -m scripts.retrain_e2_alpha99_fractional \
  --config configs/exp/e2.yaml \
  --seed 0 \
  --output-dir results/e2_fractional/0 \
  --device auto
```

After training, compare the old ceil checkpoint and new fractional checkpoint on the **same** 1,000,000 fresh QE-M paths:

```bash
python -m scripts.compare_e2_cvar_estimators \
  --old-run-dir backups/e2/0 \
  --new-run-dir results/e2_fractional/0 \
  --n-paths 1000000 \
  --batch-size 25000 \
  --evaluation-seed 10001 \
  --device auto
```

At (N=1,000,000) and (\alpha=0.99), the test-set tail contains exactly 10,000 observations. Therefore ceil and fractional CVaR evaluation are numerically identical on a fixed loss vector. Any old-versus-new difference in the 1M comparison is attributable to the **trained policy**, not evaluation rounding.

## 8. Final out-of-sample evaluation policy

A2 will use 1,000,000 fresh paths for formal E2/E3 tail-risk reporting, evaluated in chunks (default 25,000 paths) to keep GPU memory bounded.

This improves the stability of 99%-CVaR estimation: 1M paths provide 10,000 observations in the worst 1% tail rather than 1,000 under a 100k evaluation.

This does **not** change training noise. Training still uses batch 1024, so an (\alpha=0.99) update is driven by an effective tail mass of 10.24 observations.

For development and smoke tests, 100k evaluation remains acceptable. Final tables/figures should use the 1M common-path evaluation after the methodology is frozen.

## 9. Multi-seed rule and completed formal runs

Formal E2 uses seeds 0–4 with an identical frozen configuration per seed. The seed-0 fractional comparison was reviewed before the remaining formal runs began; seeds 1–4 and their one-million-path evaluations are now complete.

Fractional CVaR is retained for all formal alpha=.99 E2/E3 runs and alpha=.95/.99 call-spread runs. E1 alpha=.50 did not require retraining because its tail mass is exactly 512 at batch 1024.

The completed E2 five-seed mean prices are 1.9432 ± 0.0007 for alpha=.50 and 2.6595 ± 0.0053 for alpha=.99. The risk-ordering checks are stable, but the alpha=.99 result remains below the paper-target band [3.2, 3.8]. This is frozen as a quantitative discrepancy rather than tuned away.

## 10. Completed A2 sequence

1. Completed the seed-0 ceil-versus-fractional controlled comparison.
2. Froze the fractional CVaR implementation.
3. Completed E2 seeds 0–4 and final 1M common-path evaluations.
4. Implemented and completed the E2 call-spread experiment for alpha=.95 and .99 across seeds 0–4.
5. Completed E3 at epsilon=.01 and alpha=.99 across seeds 0–4.
6. Completed E4-BS and E4-Heston on the five-point paper epsilon grid across seeds 0–4.
7. Retained the low-friction E4 extension as optional rather than a prerequisite for freezing the main replication.

The authoritative seed registry is `docs/experiments-e2-e4.md`.

## 11. E4 five-seed asymptotic analysis and A2 figure pipeline

E4 is complete for seeds 0–4 in both Black–Scholes (d=1) and Heston (d=2). Full provenance is stored in `backups/e4bs/seed_<seed>_epsilon_summary.json` and `backups/e4h/seed_<seed>_epsilon_summary.json`; the consolidated analysis is in `docs/e4_robustness_analysis.md`.

Across independently fitted seed-level regressions, the full-grid log-log slope is 0.7129 ± 0.0017 for Black–Scholes and 0.6469 ± 0.0173 for Heston. The theoretical continuous-time exponent is 2/3. Black–Scholes is exceptionally stable. Heston seeds 0, 1, 2 and 4 cluster near .655, while seed 3 is lower at .6160 because its small-epsilon increments remain higher. Seed 3 passes the engineering and monotonicity checks and is retained as genuine seed sensitivity.

The A2 plots are independent of model training. Use `scripts/make_figures_a2.py` as the shared entry point; it dispatches to the specialist E2, E2cs, E3 and E4 plotting modules. E3 Figure 5 plotting is in `scripts/make_e3_figures.py`. E4 plots and sensitivity CSVs come directly from the saved epsilon summary JSON files, without re-running simulations.

E4 is frozen. No further retraining or seed exclusion will be used to force the fitted slope toward 2/3.

## 12. E3 final five-seed result

The frozen QE-M/current-input E3 experiment is complete for seeds 0–4. Each seed trains both recurrent and simpler policies for 30,000 iterations and uses independent one-million-path calibration and OOS samples.

| Metric | Recurrent | Simpler | Difference |
|---|---:|---:|---:|
| Indifference price | 4.1646 ± 0.0056 | 4.2225 ± 0.0040 | 0.0579 ± 0.0039 |
| OOS raw-loss CVaR99 | 4.1646 ± 0.0059 | 4.2219 ± 0.0033 | 0.0572 ± 0.0049 |
| Raw mean loss | 2.8419 ± 0.0161 | 2.9236 ± 0.0076 | 0.0817 ± 0.0119 |
| Mean transaction cost | 1.1477 ± 0.0161 | 1.2292 ± 0.0075 | 0.0815 ± 0.0120 |

The recurrent policy is better on price, raw-loss CVaR, raw mean loss and mean transaction cost in all five seeds. The raw mean-loss difference is almost entirely a transaction-cost difference. This supports the qualitative transaction-cost mechanism but not the paper's reported magnitude: the paper price gap is 1.2309, compared with 0.0579 in the baseline.

Three seed-0 fidelity controls isolate simulator and input effects. Paper inputs move both prices substantially toward the paper, while the paper-style simulator alone changes little. However, the architecture gap remains small in every arm. The five-seed baseline is retained as the formal robustness result and the controls are reported as sensitivity evidence. See `docs/e3_multiseed_summary.md` for complete results, provenance and interpretation.

## 13. Final A2 freeze decision

| Experiment | Final interpretation |
|---|---|
| E2 core | Stable qualitative risk-preference result; alpha=.50 price reproduced, alpha=.99 absolute price below the paper target |
| E2 call spread | Alpha=.99 improves CVaR99 in 5/5 seeds; hedge shape is qualitatively flatter/left-shifted but seed and BatchNorm sensitive |
| E3 | Recurrent is lower-cost/lower-price in 5/5 seeds; the direction is reproduced but the paper's large architecture gap is not |
| E4-BS | Strong cross-seed sublinear scaling, with fitted exponent consistently above 2/3 on the finite grid |
| E4-Heston | Approximate two-thirds-type scaling with greater low-epsilon seed sensitivity |

E2–E4 are frozen as of 2026-11-21. “Frozen” means no additional training solely to improve paper agreement. It does not convert partial numerical reproduction into a pass. Seed-level results, source commits and artifact locations are indexed in `docs/experiments-e2-e4.md`.

Large pathwise `pnl.npz` arrays are intentionally excluded from Git to keep the repository practical to clone. Compact metrics, resolved configurations, logs, reports and canonical figures remain version controlled.
