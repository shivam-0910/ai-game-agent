# Source of truth:
# docs/phase_1_2_technical_specification.md
#
# Based on:
# src/agent/dqn_agent.py
# src/environment/snake_env.py
#
# Phase: 5 — DQN Evaluation

"""Evaluation pipeline for a trained DQN Snake checkpoint.

Answers the Phase 5 question: "How well does the trained DQN actually
play Snake when it is evaluated without exploration or further
learning?" (Section 11 of the technical spec).

This module is deliberately independent of the training loop
(``src/training/train.py``). It reuses ``SnakeEnv`` and ``DQNAgent``
directly, but never calls ``agent.learn()``, ``agent.remember()``,
``agent.decay_epsilon()``, or ``agent.update_target_network()`` — the
loaded model is treated as frozen for the entire run (Section 4/14 of
the Phase 5 instructions).

Greedy action selection is achieved via the agent's own
``select_action(state, training=False)`` hook (see
``DQNAgent.select_action``), which already ignores epsilon and always
returns ``argmax(Q(state))``. No epsilon value is mutated on the agent
to accomplish this, so the checkpoint's own epsilon (used only for
resuming training later) is left untouched too.

Seed strategy (Section 7 of the Phase 5 instructions): each evaluation
episode receives ``base_seed + episode_index`` (episode_index starting
at 0), passed to ``env.reset(seed=...)``. This varies food placement
across episodes while remaining fully reproducible for a given
``base_seed``. The default ``base_seed`` (1_000_000) is chosen to be
far outside the range of per-episode seeds ever used during Phase 4
training (``config.seed=42`` + up to 500 => max training seed 542), so
evaluation exercises a distinct seed space from training, per Section
11 of the spec ("a separate evaluation seed set distinct from training
seeds where practical").
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

from src.agent.dqn_agent import DQNAgent
from src.environment.snake_env import SnakeEnv

# Baseline evaluation episode count. The spec (Section 11) suggests a
# range of "20-50" evaluation episodes for periodic in-training
# evaluation; for this standalone Phase 5 baseline evaluation of the
# final checkpoint we use 100, which is within the spirit of that
# guidance (more episodes narrow the confidence in the reported
# average/std-dev) while still being fast on CPU. This is NOT a value
# taken verbatim from the spec — see the Phase 5 report's Deviations
# section.
DEFAULT_NUM_EPISODES = 100

# Chosen to sit well outside the range of per-episode seeds used during
# Phase 4 baseline training (config.seed=42, 500 episodes => training
# seeds 43..542), so evaluation uses a seed space distinct from
# training per Section 11 of the spec.
DEFAULT_BASE_SEED = 1_000_000

DEFAULT_RESULTS_DIR = "results/evaluation"


@dataclass
class EvaluationConfig:
    """Evaluation-run settings.

    Only evaluation-specific values live here. DQN hyperparameters
    remain owned by ``DQNAgent`` (restored from the checkpoint via
    ``DQNAgent.load()``); environment behavior remains owned by
    ``SnakeEnv``.
    """

    checkpoint_path: str = "models/best_model.pth"
    num_episodes: int = DEFAULT_NUM_EPISODES
    base_seed: int = DEFAULT_BASE_SEED
    board_size: int = 20
    max_episode_steps: Optional[int] = None
    results_dir: str = DEFAULT_RESULTS_DIR
    verbose: bool = False


@dataclass
class EpisodeResult:
    """Result of a single greedy evaluation episode."""

    episode: int
    score: int
    reward: float
    steps: int
    snake_length: int
    termination_reason: Optional[str]
    seed: int


@dataclass
class EvaluationSummary:
    """Aggregate metrics across all evaluation episodes.

    Field names/values map directly to Section 11 of the technical
    spec (average score, maximum score, average reward, survival
    steps, performance consistency / std. dev of score) plus a couple
    of clearly-labelled supplementary fields.
    """

    num_episodes: int
    checkpoint_path: str
    base_seed: int

    average_score: float
    max_score: int
    min_score: int  # supplementary: not explicitly named in Section 11
    score_std_dev: float  # "performance consistency" per Section 11

    average_reward: float
    average_steps: float  # "survival steps (average episode length)"
    average_snake_length: float  # supplementary

    termination_counts: dict[str, int]
    termination_percentages: dict[str, float]  # supplementary

    def to_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        return d


def _greedy_episode(
    env: SnakeEnv, agent: DQNAgent, episode_number: int, seed: int, verbose: bool = False
) -> EpisodeResult:
    """Run one episode with a fixed, frozen policy and no learning.

    Mirrors the structure of ``src.training.train.run_episode`` but
    strips out everything training-related: no ``agent.remember()``,
    no ``agent.can_learn()`` / ``agent.learn()``, no
    ``agent.decay_epsilon()``. Action selection uses
    ``training=False`` so the call is unconditionally greedy
    regardless of the checkpoint's stored epsilon value.
    """
    state, _info = env.reset(seed=seed)
    terminated = False
    truncated = False

    total_reward = 0.0
    steps = 0
    info: dict = {}

    while not (terminated or truncated):
        action = agent.select_action(state, training=False)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    result = EpisodeResult(
        episode=episode_number,
        score=info["score"],
        reward=total_reward,
        steps=steps,
        snake_length=info["snake_length"],
        termination_reason=info["termination_reason"],
        seed=seed,
    )

    if verbose:
        print(
            f"Episode {episode_number}: score={result.score} steps={result.steps} "
            f"termination={result.termination_reason}"
        )

    return result


def load_agent_for_evaluation(checkpoint_path: str | Path, env: SnakeEnv) -> DQNAgent:
    """Construct a ``DQNAgent`` matching ``env`` and load a checkpoint.

    Raises
    ------
    FileNotFoundError
        If ``checkpoint_path`` does not exist. Evaluation must never
        silently fall back to a fresh, untrained model (Section 3 of
        the Phase 5 instructions) — an evaluation of an untrained
        network would be meaningless and misleading if mistaken for a
        real result.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Evaluation checkpoint not found: {checkpoint_path}. "
            "Phase 5 evaluates an already-trained model and does not "
            "train one; run Phase 4 training first or point "
            "EvaluationConfig.checkpoint_path at an existing checkpoint."
        )

    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=int(env.action_space.n),
    )
    agent.load(checkpoint_path)
    return agent


def run_evaluation(
    config: EvaluationConfig, agent: Optional[DQNAgent] = None
) -> tuple[list[EpisodeResult], EvaluationSummary]:
    """Run the full evaluation protocol described by ``config``.

    Parameters
    ----------
    config:
        Evaluation-run settings.
    agent:
        An already-loaded ``DQNAgent`` to evaluate. If omitted, an
        agent is constructed and loaded from ``config.checkpoint_path``
        via :func:`load_agent_for_evaluation`. Supplying an agent
        directly is mainly useful for tests that want to assert the
        checkpoint's weights are unmodified by comparing before/after.

    Returns
    -------
    (episode_results, summary)
    """
    env = SnakeEnv(board_size=config.board_size, max_episode_steps=config.max_episode_steps)

    if agent is None:
        agent = load_agent_for_evaluation(config.checkpoint_path, env)

    # Evaluation is inference-only: put both networks in eval mode so
    # any train-time-only layers (dropout/batchnorm — none currently
    # exist in DQNNetwork, but this keeps the intent explicit and the
    # pipeline correct if the architecture ever changes) behave
    # deterministically. DQNAgent.select_action already restores
    # policy_network.train() internally after each greedy forward pass
    # (see DQNAgent._greedy_action), so no state is disturbed by this.
    agent.policy_network.eval()

    episode_results: list[EpisodeResult] = []
    for i in range(config.num_episodes):
        episode_number = i + 1
        seed = config.base_seed + i
        result = _greedy_episode(env, agent, episode_number, seed, verbose=config.verbose)
        episode_results.append(result)

    env.close()

    summary = _aggregate(episode_results, config)
    return episode_results, summary


def _aggregate(results: list[EpisodeResult], config: EvaluationConfig) -> EvaluationSummary:
    scores = [r.score for r in results]
    rewards = [r.reward for r in results]
    steps = [r.steps for r in results]
    snake_lengths = [r.snake_length for r in results]

    termination_counts: dict[str, int] = {}
    for r in results:
        reason = r.termination_reason or "unknown"
        termination_counts[reason] = termination_counts.get(reason, 0) + 1

    n = len(results)
    termination_percentages = {
        reason: (count / n) * 100.0 for reason, count in termination_counts.items()
    }

    return EvaluationSummary(
        num_episodes=n,
        checkpoint_path=str(config.checkpoint_path),
        base_seed=config.base_seed,
        average_score=statistics.fmean(scores),
        max_score=max(scores),
        min_score=min(scores),
        # stdev requires at least 2 data points; a single-episode
        # evaluation has no defined spread.
        score_std_dev=statistics.stdev(scores) if n > 1 else 0.0,
        average_reward=statistics.fmean(rewards),
        average_steps=statistics.fmean(steps),
        average_snake_length=statistics.fmean(snake_lengths),
        termination_counts=termination_counts,
        termination_percentages=termination_percentages,
    )


# ---------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------

_RESULT_FIELDNAMES = [f.name for f in fields(EpisodeResult)]


def write_episode_results_csv(results: list[EpisodeResult], path: str | Path) -> Path:
    """Write per-episode results to a CSV file, one row per episode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RESULT_FIELDNAMES)
        writer.writeheader()
        for r in results:
            row = {name: getattr(r, name) for name in _RESULT_FIELDNAMES}
            row["termination_reason"] = row["termination_reason"] or ""
            writer.writerow(row)
    return path


def write_summary_json(summary: EvaluationSummary, path: str | Path) -> Path:
    """Write the aggregate evaluation summary as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)
    return path


@dataclass
class EvaluationRunOutput:
    """Everything produced by :func:`evaluate`, for programmatic use by
    tests/callers in addition to the files written to disk."""

    episode_results: list[EpisodeResult]
    summary: EvaluationSummary
    results_csv_path: Path
    summary_json_path: Path


def evaluate(config: EvaluationConfig, agent: Optional[DQNAgent] = None) -> EvaluationRunOutput:
    """Run evaluation and persist both output artifacts.

    This is the single top-level entry point intended for a CLI /
    notebook / other caller: it runs :func:`run_evaluation` and writes
    both ``evaluation_results.csv`` and ``evaluation_summary.json``
    under ``config.results_dir``.
    """
    episode_results, summary = run_evaluation(config, agent=agent)

    results_dir = Path(config.results_dir)
    results_csv_path = write_episode_results_csv(
        episode_results, results_dir / "evaluation_results.csv"
    )
    summary_json_path = write_summary_json(summary, results_dir / "evaluation_summary.json")

    return EvaluationRunOutput(
        episode_results=episode_results,
        summary=summary,
        results_csv_path=results_csv_path,
        summary_json_path=summary_json_path,
    )
