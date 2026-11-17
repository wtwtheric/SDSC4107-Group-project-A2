# E4 epsilon-subset slope sensitivity

Computed from saved epsilon summaries, never from retraining or simulated new samples.
Slopes are per seed; n=3 subsets are descriptive, not robust asymptotic estimators.

| Market | Seed | Subset | Slope | R² |
|---|---:|---|---:|---:|
| e4bs | 0 | all_5 | 0.7111 | 0.9991 |
| e4bs | 0 | smallest_3 | 0.6742 | 0.9995 |
| e4bs | 0 | largest_3 | 0.7458 | 0.9999 |
| e4bs | 0 | smallest_4 | 0.6939 | 0.9993 |
| e4bs | 0 | largest_4 | 0.7304 | 0.9997 |
| e4h | 0 | all_5 | 0.6534 | 0.9981 |
| e4h | 0 | smallest_3 | 0.6042 | 0.9990 |
| e4h | 0 | largest_3 | 0.6994 | 0.9998 |
| e4h | 0 | smallest_4 | 0.6308 | 0.9984 |
| e4h | 0 | largest_4 | 0.6791 | 0.9993 |

Theoretical exponent: 2/3 ≈ 0.6667. The fixed-slope reference
intercept is fitted separately; it is not an independently estimated exponent.
