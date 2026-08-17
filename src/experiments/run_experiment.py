# Source of truth:
# docs/phase_1_2_technical_specification.md
#
# Based on:
# src/agent/dqn_agent.py
# src/agent/dqn_network.py
# src/environment/snake_env.py
# src/training/train.py
# src/training/logger.py
# src/evaluation/evaluate.py
#
# Phase: 6 — Optimization and Experiments

"""Runs one controlled, single-variable experiment end to end:

    build SnakeEnv (+ optional reward_overrides)
    -> build DQNAgent (+ optional agent_overrides / network_hidden_size)
    -> train for experiment.training_episodes (mirrors src.training.train.train)
    -> save checkpoint to experiment.model_path
    -> evaluate the checkpoint with the Phase 5 evaluator, same protocol
       as the baseline (same evaluation_episodes / evaluation_base_seed)
    -> append one row to results/experiments/experiment_registry.csv

Deliberately reimplements the training loop's episode-by-episode
structure (rather than importing ``src.training.train.train``)
because that function always constructs its own default ``SnakeEnv``
with no reward-override hook and always builds a default-hyperparameter
``DQNAgent`` unless one is passed in. To keep Phase 4's tested training
module completely untouched (per the Phase 5/6 architectural
separation rule), this module reuses the same lower-level building
blocks -- ``run_episode``, ``set_global_seed``, ``EpisodeStats``,
``TrainingLogger`` -- exactly as ``train()`` does, just wired to a
per-experiment ``SnakeEnv``/``DQNAgent``.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

from src.agent.dqn_agent import DQNAgent
from src.agent.dqn_network import DQNNetwork, INPUT_SIZE, OUTPUT_SIZE
from src.environment.snake_env import SnakeEnv
from src.evaluation.evaluate import (
    EvaluationConfig,
    evaluate as run_phase5_evaluation,
    run_evaluation as run_phase5_run_evaluation,
    write_episode_results_csv,
    write_summary_json,
)
from src.experiments.experiment_config import ExperimentConfig
from src.training.logger import TrainingLogger
from src.training.train import run_episode, set_global_seed

DEFAULT_REGISTRY_PATH = "results/experiments/experiment_registry.csv"

_REGISTRY_FIELDNAMES = [
    "experiment_id",
    "category",
    "parameter",
    "baseline_value",
    "experiment_value",
    "seed",
    "training_episodes",
    "evaluation_episodes",
    "is_validation_run",
    "average_score",
    "max_score",
    "min_score",
    "score_std",
    "average_reward",
    "average_steps",
    "average_snake_length",
    "self_collision_rate",
    "wall_collision_rate",
    "status",
]


@dataclass
class ExperimentRunResult:
    """Everything produced by running one experiment, for programmatic
    inspection in addition to what gets written to disk."""

    experiment_id: str
    training_episodes_completed: int
    model_path: Path
    training_log_path: Path
    evaluation_results_csv: Path
    evaluation_summary_json: Path
    average_score: float
    max_score: int
    min_score: int
    score_std: float
    average_reward: float
    average_steps: float
    average_snake_length: float
    termination_counts: dict[str, int]
    elapsed_seconds: float


def _build_env(experiment: ExperimentConfig, board_size: int = 20) -> SnakeEnv:
    """Construct the SnakeEnv for this experiment, applying
    reward_overrides only when the experiment actually specifies them
    (so non-reward experiments get byte-for-byte baseline reward
    behavior, keeping their comparison to the baseline valid)."""
    kwargs = {"board_size": board_size}
    if experiment.reward_overrides:
        kwargs["reward_overrides"] = experiment.reward_overrides
    return SnakeEnv(**kwargs)


def _build_agent(experiment: ExperimentConfig, env: SnakeEnv) -> DQNAgent:
    """Construct the DQNAgent for this experiment, applying
    agent_overrides / network_hidden_size only when specified, so
    reward-only experiments get byte-for-byte baseline DQN
    hyperparameters."""
    agent_kwargs = dict(
        state_size=env.observation_space.shape[0],
        action_size=int(env.action_space.n),
        seed=experiment.seed,
    )
    if experiment.agent_overrides:
        agent_kwargs.update(experiment.agent_overrides)

    if experiment.network_hidden_size is not None:
        # DQNAgent doesn't expose hidden_size directly; build the
        # agent normally, then swap in differently-sized networks
        # before any training happens, so the swap is invisible to
        # everything downstream (optimizer is rebuilt to match).
        agent = DQNAgent(**agent_kwargs)
        import torch  # local import: only needed for this rare path

        hidden = experiment.network_hidden_size
        agent.policy_network = DQNNetwork(
            INPUT_SIZE, hidden_size=hidden, output_size=OUTPUT_SIZE
        ).to(agent.device)
        agent.target_network = DQNNetwork(
            INPUT_SIZE, hidden_size=hidden, output_size=OUTPUT_SIZE
        ).to(agent.device)
        agent.target_network.load_state_dict(agent.policy_network.state_dict())
        agent.target_network.eval()
        agent.optimizer = torch.optim.Adam(
            agent.policy_network.parameters(),
            lr=agent_kwargs.get("learning_rate", 1e-3),
        )
        return agent

    return DQNAgent(**agent_kwargs)


def _train_experiment(experiment: ExperimentConfig) -> tuple[Path, Path, int, float]:
    """Train one experiment's agent, mirroring src.training.train.train
    but with a per-experiment env/agent. Returns
    (model_path, training_log_path, episodes_completed, elapsed_seconds).
    """
    set_global_seed(experiment.seed)

    env = _build_env(experiment)
    agent = _build_agent(experiment, env)

    model_path = Path(experiment.model_path)
    results_dir = Path(experiment.results_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "training_log.csv"

    best_score = -1
    start_time = time.time()

    with TrainingLogger(log_path) as logger:
        for episode in range(1, experiment.training_episodes + 1):
            episode_seed = experiment.seed + episode
            stats = run_episode(env, agent, episode_seed=episode_seed)
            stats.episode = episode
            logger.log(stats)

            if stats.score > best_score:
                best_score = stats.score
                agent.save(model_path)

    env.close()

    # If no episode ever improved on -1 (shouldn't happen; score >= 0
    # always), still guarantee a checkpoint exists so evaluation has
    # something to load.
    if not model_path.exists():
        agent.save(model_path)

    return model_path, log_path, experiment.training_episodes, time.time() - start_time


def _evaluate_experiment(experiment: ExperimentConfig, model_path: Path):
    """Evaluate this experiment's checkpoint using the Phase 5
    evaluator (same greedy, no-learning protocol as the baseline).

    For architecture experiments (``network_hidden_size`` set), the
    evaluator's default DQNAgent construction would fail to load a
    non-default-width checkpoint (shape mismatch), so a matching
    agent is built here and passed in explicitly -- reusing
    ``run_evaluation``'s existing ``agent=`` hook rather than modifying
    the Phase 5 evaluation module.
    """
    eval_config = EvaluationConfig(
        checkpoint_path=str(model_path),
        num_episodes=experiment.evaluation_episodes,
        base_seed=experiment.evaluation_base_seed,
        board_size=20,
        results_dir=experiment.results_dir,
    )

    if experiment.network_hidden_size is None:
        return run_phase5_evaluation(eval_config)

    # Build a correctly-shaped agent and load the checkpoint manually,
    # then run/write via the same Phase 5 helpers evaluate() uses
    # internally, so output format is identical either way.
    env = SnakeEnv(board_size=eval_config.board_size)
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=int(env.action_space.n),
    )
    hidden = experiment.network_hidden_size
    agent.policy_network = DQNNetwork(
        INPUT_SIZE, hidden_size=hidden, output_size=OUTPUT_SIZE
    ).to(agent.device)
    agent.target_network = DQNNetwork(
        INPUT_SIZE, hidden_size=hidden, output_size=OUTPUT_SIZE
    ).to(agent.device)
    agent.load(model_path)
    env.close()

    episode_results, summary = run_phase5_run_evaluation(eval_config, agent=agent)
    results_dir = Path(eval_config.results_dir)
    results_csv_path = write_episode_results_csv(
        episode_results, results_dir / "evaluation_results.csv"
    )
    summary_json_path = write_summary_json(summary, results_dir / "evaluation_summary.json")

    from src.evaluation.evaluate import EvaluationRunOutput

    return EvaluationRunOutput(
        episode_results=episode_results,
        summary=summary,
        results_csv_path=results_csv_path,
        summary_json_path=summary_json_path,
    )


def run_experiment(experiment: ExperimentConfig) -> ExperimentRunResult:
    """Train and evaluate one experiment end to end. Does not write to
    the registry -- see :func:`run_experiment_and_record`."""
    model_path, training_log_path, episodes_completed, elapsed = _train_experiment(experiment)
    eval_output = _evaluate_experiment(experiment, model_path)
    summary = eval_output.summary

    return ExperimentRunResult(
        experiment_id=experiment.experiment_id,
        training_episodes_completed=episodes_completed,
        model_path=model_path,
        training_log_path=training_log_path,
        evaluation_results_csv=eval_output.results_csv_path,
        evaluation_summary_json=eval_output.summary_json_path,
        average_score=summary.average_score,
        max_score=summary.max_score,
        min_score=summary.min_score,
        score_std=summary.score_std_dev,
        average_reward=summary.average_reward,
        average_steps=summary.average_steps,
        average_snake_length=summary.average_snake_length,
        termination_counts=summary.termination_counts,
        elapsed_seconds=elapsed,
    )


def append_to_registry(
    experiment: ExperimentConfig,
    result: ExperimentRunResult,
    status: str,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> Path:
    """Append one row summarizing this experiment to the shared
    registry CSV, creating the file with a header if it doesn't exist
    yet. Never overwrites prior rows."""
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    n = sum(result.termination_counts.values()) or 1
    self_collision_rate = result.termination_counts.get("self_collision", 0) / n * 100.0
    wall_collision_rate = result.termination_counts.get("wall_collision", 0) / n * 100.0

    row = {
        "experiment_id": experiment.experiment_id,
        "category": experiment.category,
        "parameter": experiment.parameter,
        "baseline_value": experiment.baseline_value,
        "experiment_value": experiment.experimental_value,
        "seed": experiment.seed,
        "training_episodes": result.training_episodes_completed,
        "evaluation_episodes": experiment.evaluation_episodes,
        "is_validation_run": experiment.is_validation_run,
        "average_score": round(result.average_score, 4),
        "max_score": result.max_score,
        "min_score": result.min_score,
        "score_std": round(result.score_std, 4),
        "average_reward": round(result.average_reward, 4),
        "average_steps": round(result.average_steps, 4),
        "average_snake_length": round(result.average_snake_length, 4),
        "self_collision_rate": round(self_collision_rate, 2),
        "wall_collision_rate": round(wall_collision_rate, 2),
        "status": status,
    }

    file_exists = registry_path.exists()
    with open(registry_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_REGISTRY_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return registry_path


def run_experiment_and_record(
    experiment: ExperimentConfig,
    status: str = "screening",
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> ExperimentRunResult:
    """Convenience wrapper: run the experiment, then append its result
    to the registry. ``status`` is a free-text label such as
    ``"screening"``, ``"validated"``, or ``"rejected"`` -- assigned by
    the caller after comparing against the baseline/decision rule, not
    computed automatically here."""
    result = run_experiment(experiment)
    append_to_registry(experiment, result, status=status, registry_path=registry_path)
    return result
