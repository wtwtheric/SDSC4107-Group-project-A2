# E3 five-seed robustness and fidelity summary

## Scope

This document consolidates the frozen E3 baseline across training seeds 0–4 and the three seed-0 simulator/input fidelity controls.

The baseline uses:

- Heston QE-M simulation;
- current information set `[log(S/S0), t/T, S2/S2_0]`;
- proportional cost `epsilon=0.01`;
- `alpha=0.99`;
- recurrent and simpler policies with hidden width 17 and LayerNorm;
- 30,000 training iterations, batch size 1,024 and cosine learning rate;
- one million independent calibration paths and one million common OOS paths.

The recurrent and simpler policies are evaluated on the same paths. Their prices are calibrated separately using CVaR99.

## Five-seed baseline results

| Seed | Recurrent price | Simpler price | Price gap | Recurrent OOS CVaR | Simpler OOS CVaR | Recurrent mean cost | Simpler mean cost |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.170670 | 4.226091 | 0.055420 | 4.166625 | 4.223412 | 1.176304 | 1.237801 |
| 1 | 4.165103 | 4.225397 | 0.060294 | 4.164635 | 4.223485 | 1.141190 | 1.233726 |
| 2 | 4.163428 | 4.224009 | 0.060581 | 4.164787 | 4.225150 | 1.142704 | 1.231378 |
| 3 | 4.155811 | 4.216682 | 0.060871 | 4.155467 | 4.216636 | 1.140698 | 1.223912 |
| 4 | 4.168030 | 4.220183 | 0.052153 | 4.171663 | 4.220637 | 1.137677 | 1.219206 |
| **Mean ± sample SD** | **4.164608 ± 0.005646** | **4.222472 ± 0.003962** | **0.057864 ± 0.003902** | **4.164635 ± 0.005861** | **4.221864 ± 0.003341** | **1.147715 ± 0.016086** | **1.229205 ± 0.007534** |

The recurrent policy has a lower indifference price and lower raw-loss OOS CVaR in all five seeds. The raw mean loss is also lower in all five seeds:

- recurrent raw mean loss: `2.841870 ± 0.016059`;
- simpler raw mean loss: `2.923585 ± 0.007603`;
- mean difference: `0.081715 ± 0.011930`.

The mean transaction-cost difference is `0.081490 ± 0.012004`, almost identical to the raw mean-loss difference. After subtracting transaction costs, the mean-loss difference is only `0.000225 ± 0.000131`. The observed recurrent advantage is therefore driven primarily by lower transaction costs rather than a large difference in pre-cost hedging performance. About 85% of the average transaction-cost saving comes from the stock leg.

All operational checks pass. This supports a stable qualitative result: under proportional costs, the recurrent policy requires slightly less initial capital than the simpler policy.

## Why the Figure 5 distributions overlap

Figure 5 plots each policy at its own calibrated price:

[
p_{0,mathrm{model}}-L_{mathrm{raw,model}}.
]

Because each `p0` is separately calibrated using CVaR99, centering moves both left tails toward the same risk threshold. This removes much of the level difference that appears in the indifference prices. Strong histogram overlap is therefore compatible with a non-zero price difference.

The overlap is present across seeds rather than being caused by one failed training run. It should be reported as part of the result: the architecture changes the required price and transaction costs consistently, but it does not generate the strong distributional separation shown in the paper.

The displayed Mean Loss* is the provisional positive-part proxy `E[max(L_raw-p0,0)]`, not a verified definition of the paper's Mean Loss. It slightly favors the simpler policy in these runs and must not be used to claim a complete E3 reproduction.

## Comparison with the paper

| Result | Recurrent price | Simpler price | Gap |
|---|---:|---:|---:|
| Paper reference | 5.5137 | 6.7446 | 1.2309 |
| Five-seed QE-M/current baseline | 4.1646 | 4.2225 | 0.0579 |

The baseline price gap is about 4.7% of the paper's reported gap. The recurrent-better-than-simpler ordering is reproduced, but neither the absolute price levels nor the magnitude of the architectural effect are reproduced quantitatively.

The formal conclusion is therefore **partial qualitative reproduction**:

- reproduced: recurrent price is lower in 5/5 seeds;
- reproduced: recurrent raw-loss CVaR and average transaction costs are lower in 5/5 seeds;
- not reproduced: the paper's large recurrent–simpler price gap;
- not reproduced: a clearly separated pair of centered hedging-error distributions.

## Simulator/input fidelity controls

| Arm | Simulator | Input | Recurrent price | Simpler price | Gap |
|---|---|---|---:|---:|---:|
| Baseline seed 0 | QE-M | current | 4.1707 | 4.2261 | 0.0554 |
| Simulator only | paper-style SBK | current | 4.2278 | 4.2903 | 0.0625 |
| Paper input only | QE-M | `[log(S), V]` | 5.6320 | 5.8553 | 0.2234 |
| Simulator + paper input | paper-style SBK | `[log(S), V]` | 5.6836 | 5.8017 | 0.1181 |
| Paper reference | — | — | 5.5137 | 6.7446 | 1.2309 |

The simulator-only arm changes prices only slightly. The large movement toward the paper's price level is caused mainly by replacing the normalized current feature set with the paper input.

However, moving the overall price level closer is not the same as reproducing the paper's mechanism. The paper-input-only arm is slightly closer to the paper than the combined arm, while both still produce a much smaller recurrent–simpler gap. A plausible interpretation is that the raw paper features make the optimization problem harder under the otherwise frozen training recipe, increasing tail risk and prices. This may be a genuine fidelity effect, an optimization/feature-scaling effect, or a combination of both.

The controlled result is therefore:

1. E3 is highly sensitive to the network information set.
2. The simulator change alone is not the main explanation.
3. Paper inputs improve absolute-price agreement but do not recover the large architectural separation.
4. Numerical closeness should not be treated as proof of a more successful reproduction when the simpler-policy price and price gap remain far from the paper.

## Provenance and reporting decision

Seed 0 was trained at commit `feb6c7b`; seeds 1–4 record commit `f65e886`. The E3 training module did not change between these revisions, and the later QE-M/current defaults preserve the original baseline behaviour. The added fixed-validation sample restores RNG state and does not change the training-path sequence.

Seeds 1–4 record `dirty: true`. Their saved config hashes are identical and their results are highly consistent, so they remain internally comparable. This flag should nevertheless be retained as a provenance limitation rather than hidden.

E3 is frozen. No further retraining is required solely to force visual or numerical agreement with Figure 5. The report should use the five-seed baseline as the robustness result and the three seed-0 fidelity arms as a controlled sensitivity analysis.

## Report-ready summary

> Across five training seeds, the recurrent policy consistently required a lower CVaR99 indifference price than the simpler policy (4.1646 versus 4.2225), with an average gap of 0.0579 ± 0.0039. Its lower mean loss was almost entirely explained by lower transaction costs. The direction agrees with the paper, but the magnitude does not: the reported paper gap is 1.2309, and our centered hedging-error distributions largely overlap. Controlled fidelity experiments show that using the paper's input variables raises both prices toward the reported level, whereas changing the simulator alone has little effect. However, none of the variants recovers the paper's large recurrent–simpler separation. We therefore treat E3 as a stable qualitative but incomplete quantitative reproduction.
