# Phase 6.4 — Secondary Hyperparameters Experiment Plan

Source of truth: `docs/phase_1_2_technical_specification.md`, Section 12.
Prior stages: Phase 6.1 (reward), Phase 6.2 (epsilon decay), Phase 6.3 (learning rate).

**Status: experiments have NOT yet been run.** This document describes the objective, configuration, and protocol only.

## 1. Objective

Investigate secondary DQN hyperparameters (gamma, replay buffer capacity, batch size) starting from the Phase 6.1–6.3 adopted configuration. Exactly one variable is changed per experiment, per the Phase 6 single-variable-per-experiment rule.

## 2. Current adopted starting configuration

Phase 6.1–6.3 established the following adopted configuration:

| Setting | Value |
|---|---|
| `REWARD_DEATH` | -5.0 (Phase 6.1 adopted) |
| `epsilon_decay` | 0.995 (Phase 6.2 adopted) |
| `learning_rate` | 0.001 (Phase 6.3 adopted) |
| `gamma` | 0.95 (current default; **Stage 6.4 variable**) |
| `replay_capacity` | 100,000 (current default; **Stage 6.4 variable**) |
| `batch_size` | 64 (current default; **Stage 6.4 variable**) |

All Phase 6.4 experiments inherit `REWARD_DEATH = -5.0`, `epsilon_decay = 0.995`, and `learning_rate = 0.001` unchanged.

## 3. Why secondary hyperparameters are being investigated

Per the technical spec (Section 12), gamma / replay buffer size / batch size are the next candidate tuning variables after reward values, epsilon decay, and learning rate. These parameters affect the learning dynamics (discount factor, experience diversity, gradient stability) but are considered secondary to the primary parameters already tested in Phases 6.1–6.3.

## 4. Screening matrix

| ID | gamma | replay_capacity | batch_size | Description |
|---|-------|----------------|------------|-------------|
| S0_secondary_control | 0.95 | 100,000 | 64 | Control (adopted defaults) |
| S1_gamma_090 | 0.90 | 100,000 | 64 | Lower discount factor |
| S2_gamma_099 | 0.99 | 100,000 | 64 | Higher discount factor |
| S3_replay_50000 | 0.95 | 50,000 | 64 | Smaller replay buffer |
| S4_replay_200000 | 0.95 | 200,000 | 64 | Larger replay buffer |
| S5_batch_32 | 0.95 | 100,000 | 32 | Smaller batch size |
| S6_batch_128 | 0.95 | 100,000 | 128 | Larger batch size |

Every candidate differs from the control in exactly ONE parameter.

## 5. Variables held constant across S0–S6

Per the Phase 6.4 scope restriction, none of the following are changed in any Stage 6.4 experiment:

- Reward values (fixed at the Phase 6.1-adopted `REWARD_DEATH = -5.0` configuration)
- `epsilon_decay` (fixed at 0.995, Phase 6.2 adopted)
- `learning_rate` (fixed at 0.001, Phase 6.3 adopted)
- `epsilon_start` (1.0) and `epsilon_min` (0.01)
- `min_experiences` (1,000)
- `target_update_frequency` (1,000)
- Network architecture (default 256-unit hidden layers)
- Optimizer (Adam)
- State representation (11-feature observation)
- Action space (3 relative actions)
- Environment mechanics (`SnakeEnv(board_size=20)`, default `max_episode_steps`)

## 6. Screening protocol

- Training episodes: 100
- Training seed: 42
- Evaluation: Phase 5 `evaluate()` protocol, greedy policy, 100 episodes, base seed = 1,000,000
- Model selection during training: best-training-score checkpointing (same mechanism as Phase 4/6.1–6.3)
- Each of S0–S6 trains and evaluates its own independent checkpoint under `models/experiments/<experiment_id>/`

## 7. Validation protocol

Only a screening candidate (S1–S6) that shows a meaningful improvement over the S0 control should receive a full validation run:

- Training episodes: 500
- Training seed: 42
- Evaluation: same 100-episode, base-seed-1,000,000 protocol
- The validation run's result is compared against the current adopted configuration (Phase 6.1–6.3: average score 34.13 from R2 validation)

No 500-episode validation run is performed as part of this planning phase.

## 8. Decision rule

Following the same disciplined approach established in prior Phase 6 stages:

1. A screening candidate is only considered for validation if its average evaluation score shows a meaningful improvement over the S0 control at the same (100-episode) budget.
2. The improvement must not be driven by a single outlier episode — min score and average steps should move in a consistent direction alongside average/max score.
3. Score standard deviation should not become substantially worse without a stated justification.
4. A promising screening result must be confirmed by an independent 500-episode validation run before being adopted.
5. No arbitrary percentage threshold is defined; "meaningful improvement" is judged qualitatively against all criteria above.

## 9. Reproducibility

- All seven screening experiments use the same training seed (42) and the same evaluation base seed (1,000,000), matching Phase 5's established protocol.
- Each experiment's `ExperimentConfig` fully specifies its `seed`, `training_episodes`, `evaluation_episodes`, and `evaluation_base_seed`.
- Config files live under `configs/experiments/` and use the existing `ExperimentConfig.from_json` / `save_json` round-trip.

## 10. Limitations

- As with prior Phase 6 stages, no multi-seed replication is planned for the Stage 6.4 validation run; a single seed=42 run will be used.
- Screening budgets (100 episodes) are short enough that a real but small effect from a given secondary hyperparameter could be indistinguishable from noise at this stage.

## 11. Final Outcome

[Placeholder — to be populated after screening and validation runs are executed.]
