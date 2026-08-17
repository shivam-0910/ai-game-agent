# Phase 6 — Optimization / Experiment Plan

Source of truth: `docs/phase_1_2_technical_specification.md`, Sections
11 and 12.

## Baseline (Experiment 0)

Checkpoint: `models/best_model.pth` (Phase 4, 500 training episodes,
seed=42, board_size=20, default `DQNAgent`/`SnakeEnv` hyperparameters).

Baseline evaluation (Phase 5, 100 episodes, greedy policy, base
seed=1,000,000):

| Metric | Value |
|---|---|
| Average score | 29.26 |
| Max score | 59 |
| Min score | 9 |
| Score std. dev. | 10.49 |
| Average reward | 309.92 |
| Average steps | 439.68 |
| Average snake length | 32.26 |
| self_collision rate | 99% |
| wall_collision rate | 1% |

This baseline is preserved unchanged for the rest of Phase 6. No
experiment overwrites `models/best_model.pth`,
`results/training/training_log.csv`, `results/evaluation/evaluation_results.csv`,
or `results/evaluation/evaluation_summary.json`.

## Optimization order (Section 12 of the spec)

1. Reward values
2. Epsilon decay schedule
3. Learning rate
4. Gamma / replay buffer size / batch size
5. Network architecture size

Each stage changes exactly one variable per experiment, is screened at
a reduced training budget, and only a promising candidate receives a
full 500-episode validation run before being considered for adoption.

## Training / evaluation protocol (held constant across all experiments)

- Environment: `SnakeEnv(board_size=20)`, default `max_episode_steps`
  (unless the experiment's designated variable is the environment
  itself, which does not happen in Stage 6.1).
- State/action representation: unchanged (11-feature observation,
  3-action relative action space).
- Evaluation: Phase 5 `evaluate()`, greedy policy, 100 episodes, base
  seed = 1,000,000 — identical to the baseline protocol, for every
  experiment, so results are directly comparable.
- Model selection during training: best-training-score checkpointing,
  identical mechanism to Phase 4 (`src.training.train.train`), just
  reimplemented for a per-experiment `SnakeEnv`/`DQNAgent` in
  `src/experiments/run_experiment.py` so the tested Phase 4 module
  itself is never modified.

## Screening vs. validation budget

- Screening runs: **100 training episodes** (~2 minutes locally).
  Enough to compare relative direction/effect of a single-variable
  change without spending the full ~11-minute, 500-episode budget on
  every candidate.
- Validation runs: **500 training episodes**, same as the baseline,
  for any screening experiment whose average evaluation score beats
  the baseline. Only a validated result is eligible for adoption.

## Decision rule

An experiment is only considered a meaningful improvement over the
baseline if, in its **validation** run:

1. Average evaluation score is higher than the baseline's 29.26.
2. The improvement does not come from a single outlier episode (i.e.
   max score alone is not sufficient — average score across all 100
   evaluation episodes must improve).
3. Score standard deviation does not become substantially worse
   without a stated justification (a modest increase may be acceptable
   if average score improves meaningfully; a large increase with only
   a marginal average-score gain is not a clear win).
4. The result is reproducible: the validation run (same config,
   500 episodes) must independently confirm the screening run's
   direction of improvement, not just repeat the exact same seed's luck.

No arbitrary percentage threshold (e.g. "must beat baseline by X%") is
defined, per the Phase 6 instructions — "better" is judged qualitatively
against all four criteria above, not a single number.

## Stage 6.1 — Reward experiments (this document's active stage)

Reward tuning is first per the spec's stated rationale: reward shaping
determines *what* the agent is incentivized to learn, and the spec's
reward-hacking warnings (Section 5) require early validation. Exactly
one reward constant is changed per experiment, via the new
`SnakeEnv(reward_overrides={...})` parameter (Phase 6 addition; default
behavior when omitted is unchanged from Phase 2-5).

| ID | Parameter | Baseline | Experimental | Rationale |
|---|---|---|---|---|
| R1 | `REWARD_FOOD` | 10.0 | 15.0 | Strengthens the dominant, non-hackable incentive (score = food eaten) without touching the shaping terms that carry oscillation risk. Directly tests whether a stronger food signal improves food-seeking. Kept modest (not e.g. 50.0) so the food/death ratio does not become so lopsided that the death penalty stops mattering. |
| R2 | `REWARD_DEATH` | -10.0 | -5.0 | Phase 5's observed 99% self_collision / 1% wall_collision termination split is treated as a *hypothesis*, not a proven cause, per the Phase 6 instructions. This experiment investigates whether an overly harsh death penalty is driving over-conservative behavior that still fails at self-avoidance, by halving the penalty's magnitude and observing whether score or termination distribution shifts. |
| R3 | `REWARD_TOWARD_FOOD` / `REWARD_AWAY_FROM_FOOD` | +0.1 / -0.1 | +0.05 / -0.05 | Directly targets the spec's named "distance-shaping exploitation" risk: a smaller shaping magnitude relative to the food reward (10.0) should reduce any incentive to oscillate toward/away from food to farm the shaping reward, while still providing a directional learning signal. |

Excluded from Stage 6.1 by design: `REWARD_STEP` (already small and
non-hackable; the spec identifies no specific risk associated with it),
and combined multi-variable reward experiments (explicitly disallowed
by the Phase 6 scope rule — one variable per experiment).

## Stages 6.2–6.5

Not started. Will only begin once Stage 6.1 has concluded with either
a selected reward configuration or a decision to retain the baseline
reward values, per the spec's staged optimization order. Each stage's
document update (if any) will be appended here or to
`docs/phase_6_experiment_results.md` once experiments for that stage
actually run — no results are written in advance.
