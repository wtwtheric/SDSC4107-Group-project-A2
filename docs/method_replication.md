# Method note: replicating §5 of Buehler–Gonon–Teichmann–Wood

*Scope: this note documents the A1 part of the SDSC4107 replication — the simulation and pricing
machinery, the hedging network, the training objective, and the reimplementation choices behind
Fig. 2–4 and the E1/E1d1 experiments. It is written to be read without the code. Numbers quoted as
"paper" are from Buehler, Gonon, Teichmann & Wood, [Deep Hedging](https://arxiv.org/abs/1802.03042),
§5 (the Heston example).*

## 1. The task in one paragraph

We price and hedge a short position in a 30-day ATM call on a stock whose dynamics follow the Heston
model. The hedger may trade the stock and a variance swap, i.e. \(d=2\) hedging instruments, at 30
discrete rebalancing dates \(t_k=k\Delta t\), \(k=0,\dots,n\) with \(n=30\) and \(\Delta t=T/n\).
A hedging strategy is a map from the information set to a position vector
\(\delta_k\in\mathbb{R}^2\). The loss of the hedged portfolio is what the hedger pays out minus what
the hedge earns,

\[
L \;=\; Z \;-\; \sum_{k=1}^{n} \delta_{k-1}\cdot (S_k - S_{k-1}) \;+\; \sum_{k=0}^{n} c\,\lVert \delta_k - \delta_{k-1}\rVert_1 \cdot S_k ,
\qquad Z = (S_T-K)^+ ,
\]

with \(\delta_{-1}=0\) (the initial trade is charged), no liquidation charge at maturity, and
\(c=\varepsilon\) the proportional transaction cost. Because \(S\) is a martingale and
\(\delta_{k-1}\) is \(\mathcal{F}_{k-1}\)-measurable, \(\mathbb{E}[L]=\mathbb{E}[Z]=q\) at zero cost —
an identity, not an approximation, and the cheapest test that no instrument in the pipeline has
drifted (it is checked on every run as `mean_loss_near_q`). The *price* of the claim is the entropic
indifference price, which for a CVaR risk measure collapses to
\(p^{(\alpha)} = \mathbb{E}[L] + \mathrm{CVaR}_\alpha(L-\mathbb{E}[L]) = \mathrm{CVaR}_\alpha(L)\).
With \(\alpha=0.5\) and zero costs, \(p^{(0.5)}\) is the paper's \(p_0\), and it must exceed
\(q\approx1.69\) because \(\mathrm{CVaR}_\alpha(L)\ge\mathbb{E}[L]\) for any \(\alpha\).

## 2. Market: simulation, pricing, and the variance swap

**Parameters** (paper §5.1): \(\kappa=1\), \(\theta=0.04\), \(\sigma=2\), \(\rho=-0.7\), \(v_0=0.04\),
\(S_0=100\), \(K=100\), \(T=30/365\), \(r=0\), no dividends. The Heston call price is
\(q = 1.69183\), which matches the paper's reported \(q\approx1.69\) — a useful first check that the
interest-rate convention is right (it is \(r=0\), not \(r>0\)).

**Simulation.** Paths come from `pfhedge.stochastic.generate_heston` (Andersen's quadratic-exponential
scheme with martingale correction) rather than a hand-written Euler scheme. We call it with
`n_steps=31` and `dt = T/30`, i.e. 31 time points including \(t_0\); the RNG is PyTorch's global
generator, so seeding `torch.manual_seed(seed)` reproduces a run exactly. Everything is float64.
Hedging *and* evaluation use the same simulator, so the only model error is the QE discretisation
(30 steps), which is the same discretisation the paper uses.

**Pricing.** The Heston characteristic function is integrated with the Carr–Madan/Lewis formula,

\[
C = S_0 - \frac{1}{\pi}\int_0^\infty \mathrm{Re}\!\left[\frac{e^{-iu\log K}\,\varphi(u-i)}{iu\,\varphi(-i)}\right]\frac{du}{u^2+1/4}.
\]

A naive Gauss–Legendre grid on \(u\in[0,U]\) converges badly because the integrand's tail decays like
\(1/u^2\) only. We therefore substitute \(u=\tan\theta\), \(\theta\in[0,\pi/2)\), with
\(\theta_{\max}\) just below \(\pi/2\): \(du=(1+u^2)d\theta\) cancels the \(1/(u^2+1/4)\) factor
exactly, and 2048 Gauss–Legendre nodes give \(2.7\times10^{-13}\) absolute agreement with
`scipy.integrate.quad_vec` at the money, \(10^{-12}\) for an ITM point, and \(1.8\times10^{-5}\)
in the far OTM corner (\(S=90, v=0.01\)). The whole \(30\times208\times118\) surface costs 146 s once
and is cached.

**The variance swap.** The paper hedges with the realised variance swap, whose price at \(t\) is the
expected total variance over the remaining life of the contract,

\[
S^2_t \;=\; \mathbb{E}^{\mathbb{Q}}\!\left[\int_0^T V_s\,ds \,\middle|\, \mathcal{F}_t\right]
= \int_0^t V_s\,ds + L(t,V_t), \qquad
L(t,v) = \frac{v-\theta}{\kappa}\left(1-e^{-\kappa(T-t)}\right) + \theta\,(T-t),
\]

so \(S^2_0=\theta T=0.0032877\), and \(S^2\) is a martingale. Two details matter and cost us a day:

1. \(S^2\) is *not* the annualised fair variance rate (\(\approx\theta=0.04\)); it is 12 times smaller,
   and using the fair-strike rate instead produces a hedging instrument with a drift. A CVaR optimiser
   will happily *arbitrage* that drift: the hedge price collapses to 1.34 instead of 1.69, and the
   paper's band \([1.80,2.10]\) becomes unreachable. Every hedge instrument's price must be a
   martingale; this is the cheapest way to notice when it is not.
2. The variance-swap leg therefore needs the chain rule through \(L\):
   \(\delta_2 = (\partial C/\partial v)\,/\,(\partial_v L(t,v))\) with
   \(\partial_v L = (1-e^{-\kappa(T-t)})/\kappa\). Multiplying instead of dividing leaves the mean
   loss at 1.69 (the swap leg is driftless) but keeps \(\mathrm{CVaR}_{50}\) at 2.83 — the position is
   roughly 14 times too small to hedge volatility risk. The tell is a loss distribution as wide as the
   unhedged one even though the mean is right.

## 3. Input features (a documented deviation)

The paper feeds the network \(I_k=(\log S^1_k, V_k)\), i.e. two features for \(d=2\), the second being
the *instantaneous* variance. In a simulation we can read \(V_k\); in a market we cannot. We therefore
freeze a market-observable feature map with \(f=3\),

\[
x_k = \left[\ \log(S_k/S_0),\quad t_k/T,\quad S^2_k / S^2_0\ \right],
\]

where the third coordinate is the *remaining-life fair variance price* defined above, normalised by
its initial value. Consequences, all deliberate:

* the network sees a **scale-free, driftless** input (the variance coordinate is a price ratio, not a
  variance level), which keeps the input distribution stationary across \(k\);
* the pure time coordinate \(t_k/T\) lets a feed-forward net modulate its behaviour by date — this is
  what makes the "simpler" (non-recurrent) architecture competitive in E1;
* the cost is that the map, while injective in \((S,v)\), is not identical to the paper's, so E1's
  \(\delta\)-surface comparison is between "recurrent net on observable features" and "model delta",
  not a bit-exact reproduction of the paper's Fig. 3.

Input-layer width is \(f+d\), hidden width \(d+15\), output width \(d\), matching the paper's
\(N_1=N_2=d+15\), \(N_3=d\) once the feature count is substituted for \(N_0=2d\). For \(d=1\) the
analogous feature map is \([\log(S_k/S_0),\, t_k/T]\).

## 4. Network

Each rebalancing date uses its own small MLP (the paper's \(F_k\), with weights *not* shared across
\(k\) — the "semi-recurrent" architecture of §5.1):

* 3 linear layers, widths \(f+d \to d+15 \to d+15 \to d\);
* batch normalisation before every activation, ReLU activations;
* the previous position \(\delta_{k-1}\) is concatenated into the input, so the map is
  \(\delta_k = F_k(x_k,\delta_{k-1})\) — the recurrence is over the position, not over a hidden state;
* \(F_0(x_0,0)\) initialises the position, and the last position is held to maturity (no liquidation
  trade, hence no expiry cost).

The "simpler" baseline of E1 is the same net with the \(\delta_{k-1}\) input dropped (a purely
feed-forward map per date), which isolates how much of the improvement comes from path dependence.

## 5. Objective and optimisation

Training minimises the empirical CVaR of the terminal loss on freshly simulated paths,

\[
\min_\theta\ \mathrm{CVaR}_\alpha\big(L(\delta^\theta)\big),\qquad \alpha=0.5,
\]

implemented as \(-\frac{1}{N}\sum_{i\in \text{worst } \alpha N} L_i\) (`pfhedge`'s
`expected_shortfall` convention: the argument is the *tail probability*, so \(\alpha=0.5\) takes the
worst half). The training target and the reported price are the same functional, evaluated on disjoint
path sets: training draws 256 fresh paths per step and keeps no fixed training set, which is the
paper's online sampling.

Optimiser: Adam, initial learning rate \(5\times10^{-3}\) decayed on a cosine schedule to
\(5\times10^{-4}\) over \(3\times10^{4}\) iterations, batch of 1024 paths, no early stopping, no
gradient clipping. Neither the schedule nor the batch size is cosmetic. At a constant
\(5\times10^{-3}\) the training CVaR sits in a limit cycle (final price 2.21 against 2.01 with the
identical batch size and seed set): once the positions have reached their scale, a fixed step is
too coarse to descend further. One iteration is one batch of 1024 paths through all 31 dates:
0.032 s on one core of a Ryzen 7 9800X3D, i.e. \(\approx\)18 minutes per network for the \(10^5\)-path
evaluation included. Five seeds run as parallel single-threaded processes (`OMP_NUM_THREADS=1`).

Evaluation uses \(10^5\) freshly simulated paths with an independent seed offset
(`seed + 10^4`), so the reported price is not the training functional replayed on its own paths.

## 6. Benchmark: the model delta hedge

The comparison hedge is the classical Heston delta hedge: hold
\(\delta_1=\partial C/\partial S\) shares and
\(\delta_2=(\partial C/\partial v)/(\partial_v L)\) variance swaps, with \(C\) the Heston call price.
We build the price surface \(C(S,v,t)\) on the 30-date grid in closed form, differentiate it with
`numpy.gradient` on \((S,v)\), and interpolate with a cubic `RegularGridInterpolator`.
Because the loss of a *delta-hedged* book still depends on the path, the benchmark's loss distribution
is obtained by simulating it, exactly as for the network.

Two implementation lessons:

* **The surface must cover the simulated tail, not the plausible one.** Heston with \(\sigma=2\) and
  \(T=30/365\) produces variance spikes above 1.6 and spot below 40 within \(10^4\) paths. Our first
  grid (\(S\in[50,200]\), \(v\le0.2\)) extrapolated \(\delta_2\) to \(10^5\) and the benchmark loss
  exploded (\(\mathrm{CVaR}_{99}=134\)). The current grid is
  \(S\in[30,160]\cup[90,110]\) (208 points, refined where it matters) and
  \(v\in[0,0.12]\cup[0.12,0.4]\cup[0.4,1.8]\) (118 points).
* **A correct benchmark must reprice at \(q\).** With \(c=0\) the delta-hedged loss has zero mean, so
  over 40 000 paths \(\mathbb{E}[L]\) must equal \(q=1.6918\). We get 1.6905; anything else means a
  wrong \(\delta_2\), a drifted swap price, or a grid artefact.

With the surface fixed, the analytic hedge prices the option at
\(\mathrm{CVaR}_{50}=1.9346\) on \(10^5\) paths, and its mean loss is \(1.6914\) against
\(q=1.69183\) — the sharper check: a delta-hedged book with zero cost must be worth the option price.
The trained network lands at \(1.9431\), i.e. 0.44% above the analytic hedge and 0.2% from the
paper's \(p_0=1.94\). We deliberately do **not** claim that the network beats model delta here: at
zero cost, with the true dynamics known and used to compute \(\delta\), a \(\delta_1\) RMSE of order
0.1 is exactly the size of the residual gap, and the paper's own E1 numbers are of the same order.
The claim this experiment supports is reproduction, not dominance.

## 7. Deviations from the paper, in one table

| item | paper §5.1 | here | why |
|---|---|---|---|
| features | \(I_k=(\log S^1_k, V_k)\), \(f=2\) | \(f=3\): \(\log S/S_0\), \(t/T\), \(S^2_k/S^2_0\) | \(V_k\) is unobservable in a market; keeps the pipeline reusable for the HK data leg |
| input/hidden widths | \(2d\), \(d+15\) | \(f+d\), \(d+15\) | follows from the feature count |
| normalisation | batch norm before every activation | layer norm | the input contains the previous position, which is output of the same net; batch statistics are those of a *moving* distribution and the train/eval mismatch compounds over 30 recursive steps (we measured \(\max|\delta_1|=8.04\) against a true 0.5). Layer norm has no train/eval gap |
| output scaling | none | \(\delta=(1,100)\odot\hat\delta\), feedback divided by the same vector | the variance-swap position lives at \(\approx500\) and the stock position at \(\approx0.7\). Adam's per-coordinate step is capped by the learning rate, so from an initial \(\delta_2\approx0\) the unscaled leg needs \(O(10^4)\) steps to reach 500. The scale is *per leg*: a single 100 on both legs amplifies the stock leg's low-signal CVaR gradient by 100 and starves the variance leg |
| variance swap price | \(S^2_t\) from (5.3) | same closed form, but accrued with the step-internal CIR conditional expectation \((v-\theta)(1-e^{-\kappa\Delta})/\kappa+\theta\Delta\) | a left rectangle leaves a drift of order \((v-\theta)\kappa\Delta^2/2\), which a CVaR optimiser arbitrages: price 1.60 < \(q\), impossible |
| learning rate | constant \(5\times10^{-3}\), 5000 it | cosine \(5\times10^{-3}\to5\times10^{-4}\), \(3\times10^{4}\) it | at a constant rate the training CVaR ends in a limit cycle (2.21 vs 2.01 at the same batch/seeds) |
| batch | 256 | 1024 | halves the noise of the sampled CVaR quantile; 4096 and 8192 buy fewer iterations per unit time and under-train the time structure |
| costs | \(c_k=\varepsilon\lVert n_i\rVert S_{i,k}\) | \(\varepsilon\lVert \delta_k-\delta_{k-1}\rVert_1 S_k\), \(\delta_{-1}=0\), no expiry cost | paper's (5.7), notional-proportional |
| pricing | QuantLib | own Carr–Madan integrator | no QuantLib in the environment; verified in B0 |
| train/eval paths | fresh paths | fresh paths, disjoint seed offsets | same idea, made explicit |

## 8. Reproducibility

* `python -m scripts.check_b0` — Heston price, variance-swap price, and a mid-life mark against
  Monte Carlo (three checks, 4.5 s; results in `results/b0/b0.json`).
* `python -m src.replication.train --config configs/exp/e1.yaml --seed k` — one seed of E1;
  artifacts land in `results/e1/k/` (`metrics.json`, `config.yaml`, `run.log`, `pnl.npz`,
  `policy.pt`, `policy_simpler.pt`).
* `python -m scripts.make_figures` — Fig. 2 (loss histograms), Fig. 3 (delta surfaces), Fig. 4
  (recurrent vs simpler per seed).
* `python -m scripts.export_policy_d1` — freezes the \(d=1\) policy into
  `src/policy/weights/heston_d1.pt` for the backtest team (B2).
* `python -m scripts.recompute_d1_delta` — recomputes the \(d=1\) \(\delta_1\) RMSE/MAE quoted in §9.2
  (the `pnl.npz` files store only losses, not positions).
* `scripts/ci.py` runs ruff (check + format) and the unit tests; each `metrics.json` records the git
  commit and a dirty flag. Both batches were rerun on 2026-11-14 from tracked trees: E1 at `c3bc2ab`
  (only documentation files uncommitted, so its JSON carries `dirty: true`; `src/` untouched) and
  E1d1 at `c6ef7da` (`dirty: false`).

## 9. Results

### 9.1 E1: Heston \(d=2\), zero cost, \(\alpha=0.5\)

Five seeds, \(3\times10^{4}\) iterations each (batch 1024, cosine schedule), evaluated on \(10^5\)
fresh paths per seed:

| seed | price \(p_0\) | \(\mathbb{E}[L]\) | \(\mathrm{CVaR}_{99}\) | simpler net | \(\delta_1\) RMSE | analytic hedge |
|---|---|---|---|---|---|---|
| 0 | 1.9438 | 1.6927 | 3.548 | 1.9437 | 0.1120 | 1.9362 |
| 1 | 1.9435 | 1.6914 | 3.518 | 1.9431 | 0.0993 | 1.9338 |
| 2 | 1.9427 | 1.6908 | 3.580 | 1.9461 | 0.1129 | 1.9344 |
| 3 | 1.9421 | 1.6901 | 3.521 | 1.9435 | 0.1073 | 1.9333 |
| 4 | 1.9431 | 1.6913 | 3.563 | 1.9443 | 0.0943 | 1.9354 |
| **mean** | **1.9431** | **1.6913** | 3.546 | 1.9441 | 0.105 | 1.9346 |

with a seed-to-seed standard deviation of \(6\times10^{-4}\), i.e. the recipe, not the seed, sets the
answer. Three checks carry the replication:

* **the mean loss must equal \(q\).** \(\mathbb{E}[L]=1.6913\) against \(q=1.69183\), a 0.03% gap.
  With \(c=0\) and a martingale hedge book this is an identity, which makes it the sharpest test we
  have: it is what exposed the variance-swap discretisation bug (price and mean both 1.60, below
  \(q\), i.e. the network was harvesting the discretisation drift rather than hedging).
* **the price must land in the paper's band** \([1.80,2.10]\): 1.9431, which is 0.2% from the
  paper's \(p_0=1.94\) and 0.44% above our analytic Heston delta hedge.
* **the simpler net must be within 3%** of the recurrent one: it is within 0.06% — see Fig. 4.

**Fig. 2** overlays the two loss distributions (network, analytic delta hedge, \(10^5\) paths × 5
seeds). They are visually indistinguishable: with the true dynamics known and both legs traded, the
vega hedge already removes most of the variance risk, and the network's only room is the \(\delta_1\)
error visible in Fig. 3. The figure is therefore an honest "we reproduce \(p_0\)", not a picture of a
large hedging benefit — at zero cost and with a correct model delta, there is none to be had.

**Fig. 3** compares the position maps at \(k=15\) on the 17 875 of \(10^5\) paths that fall in
\(S\in[93,107]\), \(v\in[0.01,0.08]\) (binned means, bins with fewer than 5 paths dropped). The
network's \(\delta_1\) matches the model's to MAE 0.021 in the binned surface (RMSE 0.117 per path);
its \(\delta_2\) reproduces the shape — a ridge along \(S\approx100\), decaying with \(|S-100|\) and
peaking at low variance — with a slightly lower peak (600 against 800). The direction of the
residual is consistent with the price gap: the network is very slightly under-hedged in the variance
leg, which costs it \(\approx0.4\%\) in \(\mathrm{CVaR}_{50}\).

**Fig. 4** contrasts the recurrent (\(\delta_{k-1}\) fed back) and simpler architectures per seed.
The gap is \([-0.1,+3.0]\times10^{-3}\) in price, i.e. ≤0.15%, directional (recurrent ≤ simpler) but
negligible. This is expected rather than a defect: the Heston state \((S_k,v_k)\) is observable to
the network and already Markov, so the previous position carries almost no information the features
do not. The paper's setting where recurrence matters is the one where the state is *not* observed —
the \(d=1\) variant below, where the variance is missing entirely.

### 9.2 E1d1: Heston \(d=1\), zero cost, \(\alpha=0.5\)

The \(d=1\) variant trades the underlying only: same world, features \(x_k=[\log(S_k/S_0),t_k/T]\),
\(f=2\), no recurrence needed to represent the payoff. Five seeds, same recipe (26–32 min each):

| seed | price \(p_0\) | \(\mathbb{E}[L]\) | \(\mathrm{CVaR}_{99}\) | simpler net | analytic \(\delta_1\) hedge |
|---|---|---|---|---|---|
| 0 | 2.5291 | 1.6923 | 8.567 | 2.5163 | 2.8363 |
| 1 | 2.5290 | 1.6923 | 8.590 | 2.5186 | 2.8454 |
| 2 | 2.5276 | 1.6907 | 8.418 | 2.5295 | 2.8355 |
| 3 | 2.5429 | 1.6999 | 8.586 | 2.5214 | 2.8399 |
| 4 | 2.5402 | 1.6990 | 8.573 | 2.5261 | 2.8344 |
| **mean** | **2.5338** | **1.6948** | 8.55 | 2.5224 | 2.8383 |

Two differences from \(d=2\) are worth stating.

* **Here the deep hedge beats model delta by 10%**: \(2.534\) against \(2.838\) in
  \(\mathrm{CVaR}_{50}\). Without the variance swap the analytic delta hedge leaves the vega exposure
  unhedged, which shows up as a fat left tail (\(\mathrm{CVaR}_{99}=8.6\) against 3.5 in \(d=2\)),
  and a network that optimises the worst half of the loss learns a different strategy rather than
  approximating the analytic delta: on 20 000 fresh paths its \(\delta_1\) has RMSE 0.24 (MAE 0.20)
  against the model \(\delta_1\) at \(k=15\), \(S\in[90,110]\) (seed 0; 0.20–0.31 across the five
  seeds, so the seed-to-seed spread is comparable to the gap itself). This is the paper's
  qualitative claim for the single-instrument
  case, reproduced — and it is the opposite of the \(d=2\) outcome, where the network merely matched
  the analytic hedge.
* **Recurrence still does not matter** (2.5224 vs 2.5338, i.e. 0.4% and in the *favourable*
  direction for the simpler net, within the 3% criterion). This is the setting where the variance is
  unobserved and \(\delta_{k-1}\) could proxy for it, so the null result is informative: the position
  history adds nothing that the time-to-maturity coordinate does not already capture at this
  maturity/cost. The paper's Fig. 4 comparison is made with costs, which is where we would expect
  the effect to appear (E3, A2's task).
* The mean-loss check is looser here by necessity: one seed averages \(10^5\) losses whose standard
  deviation is large enough that \(\mathbb{E}[L]\) deviates from \(q\) by 0.001–0.008 (0.05–0.5%),
  which is why the tolerance is 0.02 rather than the 0.005 we could use in \(d=2\).

The frozen \(d=1\) policy is exported for the backtest leg: `scripts/export_policy_d1.py` writes
`src/policy/weights/heston_d1.pt` (192 KB, seed 0, with the feature spec, market parameters, seed,
training config and evaluation metrics in its `meta`), and `src/policy/heston_d1.load()` rebuilds it
against the frozen `Policy` interface for B2.


