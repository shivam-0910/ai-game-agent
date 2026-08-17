# Phase 6.2 — Epsilon Decay Experiment Plan

Source of truth: `docs/phase_1_2_technical_specification.md`, Sections
8, 11, and 12. Prior stage: `docs/phase_6_experiment_plan.md` /
`docs/phase_6_experiment_results.md` (Stage 6.1, reward experiments).

**Status: experiments have NOT yet been run.** This document describes
the objective, configuration, and protocol only. No results appear
anywhere below — see `results/experiments/experiment_registry.csv`
after the screening runs have actually been executed.

## 1. Objective

Determine whether changing the DQN agent's epsilon-greedy exploration
decay rate (`epsilon_decay`), starting from the Phase 6.1-adopted
reward configuration, meaningfully improves evaluation performance
over the current epsilon schedule. Exactly one variable is changed per
experiment, per the Phase 6 single-variable-per-experiment rule.

## 2. Current adopted starting configuration

Phase 6.1 selected `REWARD_DEATH = -5.0` as an improvement over the
baseline (validated: average evaluation score 29.26 -> 34.13). All
Stage 6.2 experiments start from this configuration, not from the
original Phase 4 baseline:

| Setting | Value |
|---|---|
| `REWARD_FOOD` | 10.0 |
| `REWARD_DEATH` | -5.0 (Phase 6.1 adopted) |
| `REWARD_STEP` | -0.01 |
| `REWARD_TOWARD_FOOD` | 0.1 |
| `REWARD_AWAY_FROM_FOOD` | -0.1 |
| `epsilon_start` | 1.0 |
| `epsilon_decay` | 0.995 (current default; **the Stage 6.2 variable**) |
| `epsilon_min` | 0.01 |
| Network architecture, learning rate, gamma, replay capacity, batch size, min experiences, target update frequency | Unchanged from Phase 3/4 baseline |

The original Phase 4 baseline checkpoint (`models/best_model.pth`) and
the Phase 6.1 validated checkpoint
(`models/experiments/R2_reward_death_VALIDATION/`) remain untouched by
this phase.

## 3. Why epsilon decay is being investigated

Per the technical spec (Section 12), epsilon decay schedule is one of
the primary candidate tuning variables after reward values, because
exploration behavior strongly affects both how much of the state space
the agent sees during training and how quickly it commits to a
possibly-suboptimal early policy. Decaying too fast risks premature
exploitation of an undertrained policy; decaying too slow wastes
training budget on essentially-random play. This stage tests both
directions around the current default (0.995).

## 4. E0 — Control

- `epsilon_decay = 0.995` (unchanged from current default)
- Purpose: a same-budget (100-episode) reference point so E1/E2/E3 are
  compared against a fair, identically-budgeted control rather than
  against the 500-episode Phase 6.1 validation numbers, mirroring how
  R0 was used in Stage 6.1.

## 5. E1 — Faster decay

- `epsilon_decay = 0.990`
- Tests whether reducing exploration faster than the current default
  benefits learning, e.g. by allowing the agent to exploit a
  reasonable policy sooner rather than continuing to explore into
  later training episodes.

## 6. E2 — Slower decay

- `epsilon_decay = 0.997`
- Tests whether retaining exploration somewhat longer than the current
  default improves learning, e.g. by letting the agent see more of the
  state space (including risky near-collision states) before
  committing to exploitation.

## 7. E3 — Very slow decay

- `epsilon_decay = 0.999`
- Tests a substantially slower transition toward exploitation, as a
  more extreme version of E2's hypothesis, and as a check for
  diminishing or reversing returns from further slowing the decay.

## 8. Variables held constant across E0-E3

Per the Phase 6.2 scope restriction, none of the following are changed
in any Stage 6.2 experiment:

- Network architecture (11 -> 256 -> 256 -> 3, ReLU, linear output)
- Learning rate
- Gamma
- Replay buffer capacity
- Batch size
- Minimum experiences threshold
- Target network update frequency
- Optimizer (Adam)
- `epsilon_start` (1.0) and `epsilon_min` (0.01)
- Reward values (fixed at the Phase 6.1-adopted `REWARD_DEATH = -5.0`
  configuration, all other rewards at baseline)
- State representation (11-feature observation)
- Action space (3 relative actions)
- Environment mechanics (`SnakeEnv(board_size=20)`, default
  `max_episode_steps`)

## 9. Screening protocol

- Training episodes: 100
- Training seed: 42 (same per-episode-seed derivation as Phase 4/6.1:
  `episode_seed = config.seed + episode`)
- Evaluation: Phase 5 `evaluate()` protocol, greedy policy, 100
  episodes, base seed = 1,000,000 — identical to every prior Phase 5/6
  evaluation, so results are directly comparable across all stages.
- Model selection during training: best-training-score checkpointing
  (same mechanism as Phase 4/6.1, via
  `src/experiments/run_experiment.py`).
- Each of E0/E1/E2/E3 trains and evaluates its own independent
  checkpoint under `models/experiments/<experiment_id>/`; none are run
  yet as of this document.

## 10. Validation protocol

Only a screening candidate (E1, E2, or E3) that shows a meaningful
improvement over the E0 same-budget control should receive a full
validation run:

- Training episodes: 500
- Training seed: 42
- Evaluation: same 100-episode, base-seed-1,000,000 protocol
- The validation run's result is then compared against the current
  adopted configuration (Phase 6.1's `REWARD_DEATH = -5.0`, 500-episode
  validated numbers: average score 34.13), not against the original
  Phase 4 baseline (29.26), since Stage 6.2 experiments start from the
  Phase 6.1-adopted configuration.

No 500-episode validation run is performed as part of this
implementation phase.

## 11. Evaluation metrics

Same metrics used throughout Phase 5/6, drawn from the evaluation
summary written by `src/evaluation/evaluate.py`: average score, max
score, min score, score standard deviation, average reward, average
steps, average snake length, and termination-reason breakdown
(`self_collision` / `wall_collision` rates).

## 12. Decision rule

Following the same disciplined approach established in
`docs/phase_6_experiment_plan.md` (Stage 6.1):

1. A screening candidate is only considered for validation if its
   average evaluation score shows a meaningful improvement over the
   E0 control at the same (100-episode) budget.
2. The improvement must not be driven by a single outlier episode —
   min score and average steps should move in a consistent direction
   alongside average/max score, not just one lucky run.
3. Score standard deviation should not become substantially worse
   without a stated justification; a modest increase alongside a
   larger average-score gain is an acceptable trade-off, as it was for
   R2 in Stage 6.1.
4. A promising screening result must be confirmed by an independent
   500-episode validation run before being adopted — the validation
   run must show the same direction of improvement, not merely repeat
   the screening run's seed-specific luck.
5. No arbitrary percentage threshold is defined; "meaningful
   improvement" is judged qualitatively against all criteria above, not
   a single number, matching Stage 6.1's decision rule.

This document does **not** predict or assume which of E1/E2/E3 (if
any) will show an improvement — that is determined by running the
experiments.

## 13. Reproducibility

- All four screening experiments use the same training seed (42) and
  the same evaluation base seed (1,000,000), matching Phase 5's
  established protocol.
- No additional random seeds are introduced at this stage.
- Each experiment's `ExperimentConfig` fully specifies its `seed`,
  `training_episodes`, `evaluation_episodes`, and
  `evaluation_base_seed`, so any of E0-E3 can be re-run independently
  and deterministically (subject to the same CPU-only, non-CUDA
  determinism caveats noted in Phase 4's `set_global_seed`
  documentation).
- Config files live under `configs/experiments/` and use the existing
  `ExperimentConfig.from_json` / `save_json` round-trip already
  exercised by the Phase 6 test suite.

## 14. Limitations

- As with Stage 6.1's R2 validation, no multi-seed replication is
  planned for the Stage 6.2 validation run; a single seed=42 run will
  be used, consistent with the Phase 6 instruction to avoid expanding
  scope beyond what each stage requires. "Reproducible" in the Section
  12 decision rule means "an independent longer run confirms the
  shorter run's direction," not "confirmed across multiple random
  seeds."
- Screening budgets (100 episodes) are short enough that a real but
  small effect from a given `epsilon_decay` value could be
  indistinguishable from noise at this stage, exactly as was observed
  for R1 in Stage 6.1 (no measurable effect at 100 episodes). A null
  screening result for E1/E2/E3 would not conclusively rule out an
  effect at a longer budget, only that it was not clear enough to
  justify a 500-episode validation run under the stated decision rule.
- These experiments are run on top of the Phase 6.1-adopted
  `REWARD_DEATH = -5.0` configuration rather than the original
  baseline. Any results are therefore specific to that combination,
  not to the original Phase 4 baseline reward configuration.
- No results are reported in this document. Once the screening runs
  are executed (by the user, per Section 9 above), results should be
  recorded following the same format as
  `docs/phase_6_experiment_results.md`, either as an update to this
  document or as a new `docs/phase_6_2_epsilon_decay_experiment_results.md`.

## 15. Final Outcome (Post-Experiment)

The screening and validation runs described in Sections 9-10 have been executed. Results are documented in `docs/phase_6_2_epsilon_decay_experiment_results.md`.

**Screening results (100 episodes):**
- E0 (control, epsilon_decay=0.995): 23.81 average score
- E1 (epsilon_decay=0.990): 33.21 average score — selected for validation
- E2 (epsilon_decay=0.997): 10.44 average score — rejected
- E3 (epsilon_decay=0.999): 0.31 average score — rejected

**Validation result (500 episodes):**
- E1 validation (epsilon_decay=0.990): 32.13 average score

**Decision:**
The E1 validation score of 32.13 was lower than the Phase 6.1 reference score of 34.13 (established with epsilon_decay=0.995). Therefore, epsilon_decay=0.990 was rejected, and the default epsilon_decay=0.995 remains the selected value.

**Final adopted configuration:**
No change — the Phase 6.1 configuration remains selected:
- REWARD_DEATH = -5.0
- epsilon_decay = 0.995

All experiment artifacts are preserved in `models/experiments/` and `results/experiments/`. The E1 validation checkpoint was NOT promoted to `models/best_model.pth`.
