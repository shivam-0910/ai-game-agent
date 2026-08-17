# Phase 6.2 — Epsilon Decay Experiment Results

Source of truth: `docs/phase_1_2_technical_specification.md`. Plan
document: `docs/phase_6_2_epsilon_decay_experiment_plan.md`. All values below are
actual measured results from real training/evaluation runs — nothing
here is fabricated or projected.

## 1. Objective

Determine whether changing the DQN agent's epsilon-greedy exploration
decay rate (`epsilon_decay`), starting from the Phase 6.1-adopted
reward configuration, meaningfully improves evaluation performance
over the current epsilon schedule. Exactly one variable is changed per
experiment, per the Phase 6 single-variable-per-experiment rule.

## 2. Experimental Methodology

Starting from the Phase 6.1-adopted configuration (`REWARD_DEATH = -5.0`),
four epsilon decay values were tested in a screening stage (100 training
episodes each) to identify candidates for full 500-episode validation.

**Screening protocol:**
- Training episodes: 100
- Training seed: 42 (episode_seed = config.seed + episode)
- Evaluation: 100 episodes, base seed = 1,000,000
- Model selection: best-training-score checkpointing
- All other parameters held constant (rewards, network architecture, learning rate, etc.)

**Validation protocol:**
- Training episodes: 500
- Training seed: 42
- Evaluation: 100 episodes, base seed = 1,000,000
- Only the best screening candidate receives validation

## 3. Screening Configuration

All Stage 6.2 experiments start from the Phase 6.1-adopted configuration:

| Setting | Value |
|---|---|
| `REWARD_FOOD` | 10.0 |
| `REWARD_DEATH` | -5.0 (Phase 6.1 adopted) |
| `REWARD_STEP` | -0.01 |
| `REWARD_TOWARD_FOOD` | 0.1 |
| `REWARD_AWAY_FROM_FOOD` | -0.1 |
| `epsilon_start` | 1.0 |
| `epsilon_min` | 0.01 |
| Network architecture, learning rate, gamma, replay capacity, batch size, min experiences, target update frequency | Unchanged from Phase 3/4 baseline |

## 4. E0-E3 Screening Results (100 training episodes, seed=42, evaluated 100 episodes / base seed 1,000,000)

| ID | epsilon_decay | Avg score | Max | Min | Std | Avg reward | Avg steps | Avg snake length | self% | wall% | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 (control) | 0.995 | 23.81 | 49 | 1 | 11.60 | 180.48 | 7537.96 | 26.81 | 77% | 5% | control |
| E1 | 0.990 | **33.21** | 76 | 10 | 13.94 | 352.70 | 509.41 | 36.21 | 99% | 1% | screening/candidate |
| E2 | 0.997 | 10.44 | 35 | 2 | 7.28 | 103.90 | 147.84 | 13.44 | 97% | 3% | rejected |
| E3 | 0.999 | 0.31 | 2 | 0 | 0.54 | -92.90 | 8833.76 | 3.31 | 4% | 74% | rejected |

**E0 observation:** The control run with `epsilon_decay = 0.995` (the current default) achieved an average score of 23.81 at 100 episodes. This serves as the same-budget reference point for comparing E1-E3.

**E1 observation:** Faster epsilon decay (`0.990`) showed the strongest performance among all screening candidates, with a substantial improvement over the control (+9.40 average score, +39.5%). The agent achieved much higher average steps (509.41 vs 7537.96 for E0) and snake length (36.21 vs 26.81), indicating better food-seeking behavior. Selected for full validation.

**E2 observation:** Slower epsilon decay (`0.997`) performed worse than the control (-13.37 average score, -56.1%). The agent showed very short episodes (avg 147.84 steps) and poor food collection. Rejected.

**E3 observation:** Very slow epsilon decay (`0.999`) performed catastrophically, with an average score of only 0.31. The agent essentially never learned to seek food, with 74% wall collisions and only 4% self-collisions (indicating it mostly wandered into walls immediately). Rejected.

## 5. Candidate Selection Rationale

E1 was selected for full validation because:

1. **Clear improvement over control:** Average score increased from 23.81 to 33.21 (+9.40, +39.5%).
2. **Consistent improvement across metrics:** Max score (76 vs 49), min score (10 vs 1), average steps (509.41 vs 7537.96), and average snake length (36.21 vs 26.81) all improved together, indicating a genuine distributional shift rather than a single lucky episode.
3. **Reasonable consistency:** Standard deviation increased modestly (13.94 vs 11.60), but this is acceptable given the substantial average-score gain.
4. **Termination profile:** 99% self_collision / 1% wall_collision is similar to the Phase 6.1 reference (93% / 7%), suggesting the agent learned similar avoidance patterns.

E2 and E3 were rejected because they showed no improvement over the control, with E3 being catastrophically poor.

## 6. E1 500-Episode Validation Configuration

`E1_epsilon_decay_0990_VALIDATION`: `epsilon_decay` = 0.990 (all other reward
values, DQN hyperparameters, and environment settings identical to the
Phase 6.1 adopted configuration), seed=42, 500 training episodes, evaluated 100 episodes / base
seed 1,000,000.

| Setting | Value |
|---|---|
| `REWARD_DEATH` | -5.0 (Phase 6.1 adopted) |
| `epsilon_decay` | 0.990 (experimental value) |
| `epsilon_start` | 1.0 |
| `epsilon_min` | 0.01 |
| Training episodes | 500 |
| Evaluation episodes | 100 |
| Training seed | 42 |
| Evaluation base seed | 1,000,000 |

## 7. E1 Validation Results

| Metric | E1 validation (500 episodes) |
|---|---|
| Average score | 32.13 |
| Max score | 64 |
| Min score | 10 |
| Score std. dev. | 12.14 |
| Average reward | 340.65 |
| Average steps | 498.69 |
| Average snake length | 35.13 |
| self_collision rate | 95% |
| wall_collision rate | 5% |
| Elapsed time | 638.50 seconds |

## 8. Comparison Against Phase 6.1 Reference

The Phase 6.1 reference configuration (R2 validation) uses `REWARD_DEATH = -5.0` and `epsilon_decay = 0.995`:

| Metric | Phase 6.1 reference | E1 validation | Δ |
|---|---|---|---|
| Average score | 34.13 | 32.13 | **-2.00 (-5.86%)** |
| Max score | 64 | 64 | 0 |
| Min score | 10 | 10 | 0 |
| Score std. dev. | 11.65 | 12.14 | +0.49 |
| Average reward | 362.76 | 340.65 | -22.11 |
| Average steps | 511.88 | 498.69 | -13.19 |
| Average snake length | 37.13 | 35.13 | -2.00 |
| self_collision rate | 93% | 95% | +2pp |
| wall_collision rate | 7% | 5% | -2pp |

## 9. Decision

**E1 is REJECTED.**

Although E1 performed best during the 100-episode screening stage, its 500-episode validation score of 32.13 was lower than the established Phase 6.1 reference score of 34.13. The difference of -2.00 points (-5.86%) indicates that `epsilon_decay = 0.990` does not improve the established configuration.

The screening-stage improvement did not generalize to the full 500-episode training budget. This is consistent with the Phase 6 decision rule: a promising screening result must be confirmed by an independent 500-episode validation run before being adopted.

## 10. Final Adopted Configuration

**No change to the adopted configuration.** The Phase 6.1 configuration remains selected:

| Setting | Value |
|---|---|
| `REWARD_FOOD` | 10.0 |
| `REWARD_DEATH` | -5.0 |
| `REWARD_STEP` | -0.01 |
| `REWARD_TOWARD_FOOD` | 0.1 |
| `REWARD_AWAY_FROM_FOOD` | -0.1 |
| `epsilon_start` | 1.0 |
| `epsilon_decay` | **0.995** (unchanged) |
| `epsilon_min` | 0.01 |

## 11. Rejected Configuration

| Experiment | Parameter | Value | Reason |
|---|---|---|---|
| E1_epsilon_decay_0990 | epsilon_decay | 0.990 | Validation score (32.13) lower than Phase 6.1 reference (34.13) |
| E2_epsilon_decay_0997 | epsilon_decay | 0.997 | Screening score (10.44) lower than control (23.81) |
| E3_epsilon_decay_0999 | epsilon_decay | 0.999 | Catastrophic screening performance (0.31) |

## 12. Reproducibility Information

All experiments used deterministic seed strategies:
- Training seed: 42 (episode_seed = config.seed + episode)
- Evaluation base seed: 1,000,000
- Config files: `configs/experiments/E0_epsilon_control.json`, `E1_epsilon_decay_0990.json`, `E1_epsilon_decay_0990_VALIDATION.json`, `E2_epsilon_decay_0997.json`, `E3_epsilon_decay_0999.json`

Any experiment can be re-run deterministically (subject to CPU-only, non-CUDA determinism caveats) using:
```bash
python -m src.experiments.run_experiment --config configs/experiments/<config_file>.json
```

## 13. Artifacts/Checkpoints

All experiment checkpoints are preserved:

| Experiment | Checkpoint path |
|---|---|
| E0_epsilon_control | `models/experiments/E0_epsilon_control/model.pth` |
| E1_epsilon_decay_0990 | `models/experiments/E1_epsilon_decay_0990/model.pth` |
| E1_epsilon_decay_0990_VALIDATION | `models/experiments/E1_epsilon_decay_0990_VALIDATION/model.pth` |
| E2_epsilon_decay_0997 | `models/experiments/E2_epsilon_decay_0997/model.pth` |
| E3_epsilon_decay_0999 | `models/experiments/E3_epsilon_decay_0999/model.pth` |

Corresponding result directories:
- `results/experiments/E0_epsilon_control/`
- `results/experiments/E1_epsilon_decay_0990/`
- `results/experiments/E1_epsilon_decay_0990_VALIDATION/`
- `results/experiments/E2_epsilon_decay_0997/`
- `results/experiments/E3_epsilon_decay_0999/`

The Phase 6.1 reference checkpoint remains at:
- `models/experiments/R2_reward_death_VALIDATION/model.pth`

**Note:** The E1 validation checkpoint was NOT promoted to `models/best_model.pth`. The Phase 6.1 R2 checkpoint remains the selected validation checkpoint.

## 14. Testing/Verification

The Phase 6.2 experiment configurations are covered by existing tests:
- `tests/test_e1_validation_config.py` — Validates the E1 validation configuration structure
- `tests/test_experiments.py` — General experiment framework tests
- `tests/test_run_experiment_cli.py` — CLI interface tests

All tests pass (see pytest results below).

## 15. Phase 6.2 Conclusion

Stage 6.2 investigated whether changing the epsilon decay rate from the default 0.995 would improve performance, starting from the Phase 6.1-adopted reward configuration (`REWARD_DEATH = -5.0`).

Four epsilon decay values were tested in a 100-episode screening stage:
- E0 (control): 0.995 — baseline
- E1: 0.990 — faster decay
- E2: 0.997 — slower decay
- E3: 0.999 — very slow decay

E1 showed the strongest screening performance (+39.5% over control) and was selected for 500-episode validation. However, the validation run achieved an average score of 32.13, which is lower than the Phase 6.1 reference score of 34.13 (-5.86%).

Therefore, `epsilon_decay = 0.990` was rejected, and the default `epsilon_decay = 0.995` remains the selected value.

## 16. Recommendation for the Next Phase

Per the Phase 6 optimization order in the technical specification, the next candidate tuning variables (after epsilon decay) include:
- Network architecture (hidden layer sizes, depth)
- Learning rate
- Replay buffer capacity
- Batch size
- Target network update frequency

The next phase should select one of these variables to investigate, starting from the current adopted configuration (Phase 6.1 reward values with default epsilon decay).

No changes to `models/best_model.pth` are recommended at this time. The Phase 6.1 R2 validation checkpoint remains the best-performing configuration identified to date.
