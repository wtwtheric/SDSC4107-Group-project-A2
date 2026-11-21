> **Superseded for robustness conclusions:** seed 0 remains the original pilot analysis, but the frozen five-seed conclusion is now in `docs/e4_robustness_analysis.md` and `reports/e4/e4_robustness_summary.json`.

# E4 Seed-0 Analysis — Price Asymptotics (Figures 10–11)

Status: Seed 0 analysis from archived JSON results. Seeds 1–4 remain for robustness. No checkpoint was retrained or evaluated during this analysis.

## 1. Scope and methodology

We reproduce the power-law prediction from Deep Hedging Section 5.3:

\[
p_\varepsilon-p_0 \propto \varepsilon^{2/3},
\qquad \log(p_\varepsilon-p_0)=a+b\log\varepsilon.
\]

Use epsilon values 2^-10 through 2^-6, entropic risk with lambda=1, daily hedging over 30 steps, and one independently trained recurrent policy per epsilon. The BS model uses one stock hedge instrument, while Heston uses the stock and variance swap (d=2). Training uses 30,000 iterations, batch size 1,024; each OOS estimate uses 100,000 paths.

Source of all numerical claims: backups/e4bs/seed_0_epsilon_summary.json and backups/e4h/seed_0_epsilon_summary.json on exp/a2-seed0-batch. Reference zero-cost risk-neutral prices are BS 2.2871506280 and Heston 1.6918339251. This risk-neutral baseline is *not* a separately trained entropic zero-cost hedging policy.

We preserve A1/A2's LayerNorm choice, PFHedge QE-M Heston simulation, and feature preprocessing. These are documented implementation differences from the paper.

## 2. Price increments

| Epsilon | BS increment | Heston increment |
|---:|---:|---:|
| 0.0009765625 | 0.248870 | 0.231020 |
| 0.0019531250 | 0.390274 | 0.343118 |
| 0.0039062500 | 0.633696 | 0.533826 |
| 0.0078125000 | 1.052256 | 0.856330 |
| 0.0156250000 | 1.781896 | 1.407538 |

All ten increments are positive and strictly increase with transaction costs.

## 3. Full-grid log-log regression

The five points for each market are fit by OLS in log epsilon and log price increment. Each figure includes (i) observations, (ii) a free-intercept/free-slope OLS line and (iii) a slope-fixed-at-2/3 theoretical reference line whose intercept is fitted to the same five points. The latter is a reference, **not** a separately estimated exponent.

| Result | BS | Heston |
|---|---:|---:|
| Free fitted slope | **0.7111** | **0.6534** |
| Free OLS R² | 0.9991 | 0.9981 |
| Fixed 2/3 line R² | 0.9952 | 0.9977 |
| Theoretical exponent | 0.6667 | 0.6667 |
| Paper figure reference | ~0.67 | ~0.71 |
| Pre-set acceptance interval | [0.63, 0.71] | [0.66, 0.76] |

The very high R² values support a log-log power-law pattern across the tested epsilon range. Using the pre-set numerical thresholds strictly, both slopes fall marginally outside: BS by +0.0011 relative to its upper bound, Heston by -0.0066 relative to its lower bound. Do not adjust epsilon subsets to make the slopes hit target bands.

## 4. Epsilon-subset sensitivity

| Subset | BS slope | Heston slope |
|---|---:|---:|
| All five | 0.7111 | 0.6534 |
| Smallest three | 0.6742 | 0.6042 |
| Largest three | 0.7458 | 0.6994 |
| Smallest four | 0.6939 | 0.6308 |
| Largest four | 0.7304 | 0.6791 |

BS moves closer to theoretical 2/3 when considering the smallest three costs, whereas Heston moves further away. A strong five-point R² therefore does **not** establish that the limiting asymptotic regime epsilon -> 0 has been attained. Three-point fits are especially sensitive and are exploratory diagnostics.

The scripts generate the epsilon sensitivity table (CSV and Markdown), price-increment CSV, and per-seed slope summary. For robustness, fit one slope per seed and report mean ± sample SD; do not pool 25 pairs as if the five repeated epsilons were 25 independent cost levels.

## 5. Economic interpretation and limitations

The price premium increases sublinearly with transaction costs. Both BS and Heston show an approximate two-thirds scaling, consistent with the paper's main asymptotic direction. However, inference is limited to one seed, five finite positive cost levels, 30-step discrete rebalancing, finite 100K-path OOS evaluation, possible optimizer error, and the deviations from the authors' simulation and normalization. The reference p0 is the risk-neutral price, not a separate zero-cost entropic hedge optimum; this baseline can affect a fitted exponent at small epsilon.

Assessment: strong seed-0 BS scaling pattern and substantial seed-0 Heston support, with greater Heston small-epsilon sensitivity. No claim about cross-seed uncertainty is made until seeds 1–4 are complete.

## 6. Reproducibility (no retraining)

From the repository root:

    python -m scripts.a2.make_e4_figures --data-root backups --figures-dir figures --report-dir reports/e4 --seeds 0

Or via the unified A2 figure dispatcher:

    python -m scripts.a2.make_figures_a2 --only e4 --seeds 0

Outputs are figures/fig10_e4_bs.png and figures/fig11_e4_heston.png, plus reports/e4/e4_epsilon_sensitivity.csv, e4_epsilon_sensitivity.md, e4_price_increments.csv and e4_seed_slopes.csv.

After all other seeds have been independently trained and archived, rerun with --seeds 0 1 2 3 4. The script checks each seed's epsilon grid and per-row source, and only reads the saved JSON; it cannot modify checkpoints or run Monte Carlo.
