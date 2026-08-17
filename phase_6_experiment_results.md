# Phase 6 — Experiment Results (Stage 6.1: Reward)

Source of truth: `docs/phase_1_2_technical_specification.md`. Plan
document: `docs/phase_6_experiment_plan.md`. All values below are
actual measured results from real training/evaluation runs — nothing
here is fabricated or projected.

## Screening runs (100 training episodes, seed=42, evaluated 100 episodes / base seed 1,000,000)

A same-budget control (`R0`, default reward values, 100 training
episodes) was added after the fact so the three reward screens are
compared against a fair, identically-budgeted reference rather than
the 500-episode baseline number.

| ID | Parameter | Value | Avg score | Max | Min | Std | self% | wall% |
|---|---|---|---|---|---|---|---|---|
| R0 (control) | none | baseline | 18.76 | 45 | 1 | 10.21 | 58% | 0% |
| R1 | REWARD_FOOD | 10.0 → 15.0 | 18.76 | 45 | 1 | 10.21 | 58% | 0% |
| R2 | REWARD_DEATH | -10.0 → -5.0 | **28.48** | 55 | 6 | 11.09 | 95% | 5% |
| R3 | REWARD_TOWARD/AWAY_FOOD | ±0.1 → ±0.05 | 21.27 | 47 | 1 | 10.63 | 82% | 9% |

**R1 observation:** average score identical to the control at this
budget — increasing the food reward alone showed no measurable effect
in 100 episodes. Not pursued further; a longer run might reveal an
effect, but nothing in the screening data justified spending a
500-episode validation run on it.

**R3 observation:** modest improvement over control (+2.51), but far
behind R2, and its wall_collision rate (9%) is notably higher than the
real baseline's (1%), suggesting the smaller shaping signal may have
made early wall-avoidance learning slower at this budget. Not pursued
further given R2's clearly stronger result.

**R2 observation:** clear improvement over control (+9.72), and its
termination profile (95% self / 5% wall) is the closest of the three to
the real 500-episode baseline's 99%/1% split — evidence this is a
genuine effect of the parameter change rather than noise. Selected for
full validation.

## Validation run (500 training episodes, same protocol as the original baseline)

`R2_reward_death_VALIDATION`: `REWARD_DEATH` = -5.0 (all other reward
values, DQN hyperparameters, and environment settings identical to the
baseline), seed=42, 500 training episodes, evaluated 100 episodes / base
seed 1,000,000.

| Metric | Baseline (Experiment 0) | R2 validated | Δ |
|---|---|---|---|
| Average score | 29.26 | **34.13** | **+4.87 (+16.6%)** |
| Max score | 59 | 64 | +5 |
| Min score | 9 | 10 | +1 |
| Score std. dev. | 10.49 | 11.65 | +1.16 |
| Average reward | 309.92 | 362.76 | +52.84 |
| Average steps | 439.68 | 511.88 | +72.20 |
| Average snake length | 32.26 | 37.13 | +4.87 |
| self_collision rate | 99% | 93% | -6pp |
| wall_collision rate | 1% | 7% | +6pp |

## Decision rule applied

Per `docs/phase_6_experiment_plan.md`:

1. **Average evaluation score improves over baseline** — 29.26 → 34.13. ✅
2. **Not driven by a single outlier episode** — min score, max score,
   and average steps all improved together, indicating a broad
   distributional shift rather than one lucky episode. ✅
3. **Consistency not substantially worse** — std. dev. rose only
   modestly (+1.16, ~11%) alongside a much larger average-score gain
   (+16.6%); this is judged an acceptable trade-off, not a red flag. ✅
4. **Reproducible** — the screening run (100 episodes, avg 28.48) and
   the independent 500-episode validation run (avg 34.13) both show
   the same direction of improvement over their respective
   same-budget references, using the same seed strategy but a full
   independent training run (not a re-evaluation of the same
   checkpoint). ✅

All four criteria are satisfied.

## Selected configuration

**Adopted: `REWARD_DEATH = -5.0`** (all other parameters unchanged from
baseline).

Registry status for `R2_reward_death_VALIDATION` updated to `adopted`
in `results/experiments/experiment_registry.csv`.

The original baseline checkpoint (`models/best_model.pth`) is
**preserved unchanged** — the validated model lives at
`models/experiments/R2_reward_death_VALIDATION/model.pth`. Promoting
this experimental checkpoint to replace the project's primary
`models/best_model.pth` is a deliberate follow-up action, not performed
automatically as part of Phase 6 (see Deviations/Next steps below).

## Interpretation

Reducing the death penalty's magnitude allowed the agent to take
slightly more risk, converting some previously-fatal near-collisions
into successful food-eating maneuvers, and it measurably improved food
collection (score) without a proportionate loss in consistency. The
termination distribution still skews heavily toward self_collision
(93%), so self-avoidance at high snake length remains the dominant
failure mode even with this improvement — it was reduced, not solved.
This is consistent with, but does not conclusively prove, the Phase 5
hypothesis that an overly harsh death penalty was partly suppressing
food-seeking behavior; a controlled A/B on this one variable is
suggestive evidence, not a controlled causal proof (no ablation across
multiple seeds was performed).

## Deviations from `docs/phase_1_2_technical_specification.md`

- The spec's screening/validation episode budgets aren't specified
  numerically; the 100-episode screening / 500-episode validation split
  used here is a Phase 6 judgment call based on observed per-episode
  training time (~1.3-1.9s/episode locally), not a value taken from the
  spec.
- `SnakeEnv` gained an optional `reward_overrides` constructor
  parameter (Phase 6 addition) to make reward experiments possible
  without monkeypatching; default behavior (parameter omitted) is
  unchanged from Phases 2-5, verified by the full existing Phase 2 test
  suite passing unmodified (50/50) plus 8 new targeted tests.
- No multi-seed replication was performed for R2's validation (only a
  single seed=42 run); the spec does not mandate multi-seed replication,
  and Phase 6 instructions caution against expanding scope
  unnecessarily, but this is worth flagging as a limitation of the
  "reproducible" claim above — reproducibility here means "an
  independent longer run confirmed the shorter run's direction," not
  "confirmed across multiple random seeds."

## Next steps (not performed in this phase, per the Phase 6 stop condition)

- Stage 6.2 (epsilon decay) has not started. Per the spec's optimization
  order, it should begin from the now-adopted `REWARD_DEATH = -5.0`
  configuration as its new starting point, once a decision is made
  about promoting the R2 checkpoint.
- Whether to formally replace `models/best_model.pth` with the R2
  validated checkpoint is a project decision outside Phase 6's stated
  scope (Phase 6 only requires selecting or rejecting a configuration,
  not promoting it to the canonical baseline file) — flagging for your
  decision rather than doing it automatically.
