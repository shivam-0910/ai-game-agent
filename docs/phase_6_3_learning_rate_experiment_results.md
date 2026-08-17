# Phase 6.3 — Learning Rate Experiment Results

## Screening Results

| Experiment | learning_rate | Average Score | Max | Min | Std Dev |
|------------|---------------|---------------|-----|-----|---------|
| L0_learning_rate_control | 0.001 | 23.81 | 49 | 1 | 11.5991 |
| L1_learning_rate_0005 | 0.0005 | 26.20 | 60 | 1 | 11.3511 |
| L2_learning_rate_002 | 0.002 | 30.79 | 77 | 7 | 13.5119 |
| L3_learning_rate_005 | 0.005 | 24.13 | 52 | 5 | 9.2416 |

## Validation Result

| Experiment | learning_rate | Training Episodes | Average Score | Max | Min | Std Dev |
|------------|---------------|-------------------|---------------|-----|-----|---------|
| L2_learning_rate_002_VALIDATION | 0.002 | 500 | 26.42 | 53 | 8 | 10.6431 |

## Final Decision

L2 validation scored 26.42, which is below the Phase 6.1 reference score of 34.13.

**L2 is rejected.**

The adopted configuration remains unchanged:
- REWARD_DEATH = -5.0
- epsilon_decay = 0.995
- learning_rate = 0.001
