# Phase 6.3 — Learning Rate Optimization: Experiment Plan

Source of truth: `docs/phase_1_2_technical_specification.md`
Based on: `src/agent/dqn_agent.py`, `src/experiments/experiment_config.py`,
`src/experiments/run_experiment.py`, `configs/experiments/E*.json`, `configs/experiments/R*.json`

## 1. Objective

Optimize the DQN learning rate used by the Adam optimizer, starting from the
currently adopted Phase 6.1/6.2 configuration, while holding every other
hyperparameter fixed. This is a **screening** phase: it identifies promising
candidate learning rates for a later full-budget validation run, mirroring
how Phase 6.2 screened epsilon-decay values before validating E1.

## 2. Current adopted configuration

As of the end of Phase 6.2:

| Parameter        | Adopted value | Source                                   |
|-------------------|--------------|-------------------------------------------|
| `REWARD_DEATH`     | `-5.0`       | Phase 6.1 (R2), validated in R2_VALIDATION |
| `epsilon_decay`    | `0.995`      | Phase 6.2 (E0 control); E1 (0.990) failed 500-episode validation and was rejected |
| `learning_rate`    | `0.001`      | `DQNAgent.DEFAULT_LEARNING_RATE` (`1e-3`), unchanged since Phase 3 — never previously tuned |

All Phase 6.3 experiments inherit `REWARD_DEATH = -5.0` and
`epsilon_decay = 0.995` unchanged.

## 3. Why learning rate is being tested

Learning rate has never been tuned in this project — it has only ever used
the Phase 3 baseline default (`1e-3`) that was chosen as a reasonable
starting point, not validated empirically. Learning rate directly controls
the Adam optimizer's step size and is one of the most consequential DQN
hyperparameters for both convergence speed and training stability, making it
a natural next candidate for the staged Phase 6 optimization sequence after
reward shaping (6.1) and exploration schedule (6.2).

## 4. Repository inspection findings

Before writing any code or configs, the existing implementation was
inspected directly (not assumed):

1. **Where is learning rate defined?** `src/agent/dqn_agent.py`, module-level
   constant `DEFAULT_LEARNING_RATE = 1e-3`.
2. **Current default value:** `0.001`, matching the plan's assumed L0 value —
   no silent change was needed.
3. **Does `DQNAgent` accept `learning_rate` as a constructor parameter?**
   Yes — `__init__(..., learning_rate: float = DEFAULT_LEARNING_RATE, ...)`.
4. **How is the optimizer created?**
   `self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)`
   inside `__init__`.
5. **Does `ExperimentConfig` support `agent_overrides`?** Yes — it already
   has an `agent_overrides: Optional[dict[str, Any]]` field, used by every
   Phase 6.2 epsilon-decay experiment.
6. **Does `run_experiment.py` pass `agent_overrides` into `DQNAgent`?** Yes —
   `_build_agent()` does `agent_kwargs.update(experiment.agent_overrides)`
   before constructing `DQNAgent(**agent_kwargs)`.
7. **Can `learning_rate` be changed through `ExperimentConfig` without
   modifying the training framework?** Yes, fully — `agent_overrides =
   {"learning_rate": <value>}` flows through exactly the same code path that
   already carries `epsilon_decay` overrides.

**Conclusion: no changes to `src/agent/dqn_agent.py`,
`src/experiments/experiment_config.py`, or `src/experiments/run_experiment.py`
were necessary.** The existing `agent_overrides` mechanism, built during
Phase 6.2, already generalizes to any `DQNAgent` constructor parameter,
including `learning_rate`. Phase 6.3 preparation is configuration- and
test-only.

## 5. Control experiment

**`L0_learning_rate_control`** — `learning_rate = 0.001` (the adopted
default), with `baseline_value == experimental_value == "0.001"`, mirroring
how `E0_epsilon_control` was defined in Phase 6.2. This provides a
same-protocol reference point trained under identical conditions to the
screening experiments, controlling for any run-to-run variance introduced by
the training/evaluation pipeline itself.

## 6. Screening experiments and exact learning-rate values

| ID  | learning_rate | Description                    |
|-----|---------------|---------------------------------|
| L0  | 0.001         | Control (adopted default)       |
| L1  | 0.0005        | Lower learning rate             |
| L2  | 0.002         | Higher learning rate            |
| L3  | 0.005         | Much higher learning rate       |

No repository documentation or existing Phase 6 optimization plan specifies
alternative learning-rate values, so the plan's proposed screening values
(0.001 / 0.0005 / 0.002 / 0.005) were used as given — a roughly log-spaced
sweep (0.5x, 2x, 5x the control) is a standard, reasonable first pass for
learning-rate screening.

## 7. Variables held constant

Across all four experiments (L0–L3):

- `REWARD_DEATH = -5.0`
- `epsilon_decay = 0.995`
- `gamma`, `epsilon_start`, `epsilon_min`, replay buffer capacity/size,
  batch size, target update frequency — all left at `DQNAgent` defaults
  (not overridden by any L-series config)
- Network architecture (`network_hidden_size: null` → default 256-unit
  hidden layers)
- Optimizer type (Adam)
- Training algorithm (`src/training/train.run_episode` / the
  `run_experiment.py` training loop, unchanged)
- Environment behavior (`SnakeEnv`, only `reward_overrides` applied, same as
  every non-reward Phase 6 experiment)
- `seed = 42`, `training_episodes = 100`, `evaluation_episodes = 100`,
  `evaluation_base_seed = 1_000_000` — identical to every Phase 6.2
  screening experiment

The only varying value across L0–L3 is `agent_overrides["learning_rate"]`.

## 8. Experimental budget

Same screening budget/protocol established by Phase 6.1 and Phase 6.2:

- **Training:** 100 episodes per screening experiment
- **Evaluation:** 100 episodes, greedy/no-learning, `base_seed = 1,000,000`
- **Validation (if warranted):** a full 500-episode training run of the most
  promising screening candidate, following the same pattern as
  `E1_epsilon_decay_0990_VALIDATION` and `R2_reward_death_VALIDATION`. No
  L-series validation config has been created yet — it will only be created
  once a screening result identifies a genuinely promising candidate.

## 9. Evaluation methodology

Identical to Phase 6.1/6.2: each experiment's best checkpoint (highest
training score, saved via the same `agent.save()` best-model criterion used
by `run_experiment.py`) is evaluated with the Phase 5 evaluator
(`src.evaluation.evaluate`) in greedy (no-exploration, no-learning) mode over
100 fixed evaluation episodes seeded from `evaluation_base_seed = 1,000,000`,
so every experiment — L-series included — is assessed on the exact same set
of evaluation episodes as every prior Phase 6 experiment.

## 10. Selection criteria

A screening candidate is considered promising if it shows a meaningfully
higher `average_score` (and/or lower `score_std`) than L0 in the
`experiment_registry.csv` row, without a corresponding collapse in training
stability (e.g. diverging/exploding loss visible in that experiment's
`training_log.csv`). Given Phase 6.2's experience — a screening-promising
candidate (E1) failed full validation — no learning-rate value should be
adopted from screening results alone.

## 11. Validation protocol

Exactly the Phase 6.2 pattern: whichever L-series screening experiment looks
most promising gets one full 500-episode validation run (a new
`L{n}_learning_rate_..._VALIDATION.json` config, `training_episodes: 500`,
`is_validation_run: true`) before being adopted as the new default. A
screening win alone is not sufficient grounds for adoption, per the
documented E1 outcome.

## 12. Expected output artifacts

Once experiments are actually run via the existing CLI
(`python -m src.experiments.run_experiment --config <path>`), each will
produce:

- `models/experiments/L{n}_learning_rate_.../model.pth` — best checkpoint
- `results/experiments/L{n}_learning_rate_.../training_log.csv`
- `results/experiments/L{n}_learning_rate_.../evaluation_results.csv`
- `results/experiments/L{n}_learning_rate_.../evaluation_summary.json`
- One appended row in `results/experiments/experiment_registry.csv`

**No experiments have been executed as part of this Phase 6.3 preparation
task. None of the artifacts listed above exist yet.** This document
intentionally contains no results section with numbers, because there are
none to report.

## 13. Reproducibility information

- All L-series configs use `seed = 42`, matching every prior Phase 6
  experiment, so the training-side RNG state (Python `random`, NumPy,
  PyTorch, and per-episode environment seeds derived from the master seed)
  is reproducible run-to-run.
- `evaluation_base_seed = 1_000_000` is identical across all L-series
  configs and matches every prior Phase 6 experiment, so evaluation episodes
  are directly comparable across L0–L3 and against every earlier experiment.
- Each config's `model_path` and `results_dir` are unique to that
  experiment, so re-running any individual experiment does not overwrite
  another experiment's artifacts.
- Running an experiment: `python -m src.experiments.run_experiment --config configs/experiments/L0_learning_rate_control.json`
  (and analogously for L1/L2/L3).

## 14. Final Outcome

L2 was selected for validation after screening (30.79 average score, highest among L0–L3).

L2 validation scored 26.42 over 500 training episodes.

Phase 6.1 reference score was 34.13.

L2 was rejected.

The adopted configuration remains unchanged:
- REWARD_DEATH = -5.0
- epsilon_decay = 0.995
- learning_rate = 0.001
