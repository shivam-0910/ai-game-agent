# Source of truth:
# docs/phase_1_2_technical_specification.md
#
# Based on:
# src/agent/dqn_agent.py
# src/environment/snake_env.py
#
# Phase: 4 — DQN Training Pipeline

"""DQN training loop for the Snake environment.

Orchestrates the existing ``SnakeEnv`` and ``DQNAgent`` components
(Section 6 of the Phase 4 instructions: this module does not
reimplement the Bellman equation, the optimizer, the replay buffer, or
target-network logic — all of that is owned by ``DQNAgent``).

Episode structure (Section 3/4):

    reset environment
    while not (terminated or truncated):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.remember(state, action, reward, next_state, terminated or truncated)
        state = next_state
        if agent.can_learn():
            loss = agent.learn()
    agent.decay_epsilon()
    record episode statistics
    checkpoint if appropriate
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.agent.dqn_agent import DQNAgent
from src.environment.snake_env import SnakeEnv
from src.training.config import TrainingConfig
from src.training.logger import TrainingLogger, EpisodeStats


def set_global_seed(seed: int) -> None:
    """Seed Python ``random``, NumPy, and PyTorch for reproducibility.

    The environment's own RNG is seeded separately via
    ``env.reset(seed=...)`` (Gymnasium convention), not here. Per
    Section 9 of the Phase 4 instructions, this does not claim perfect
    determinism (CUDA nondeterminism, if a GPU is later used, is out of
    scope for this CPU-only baseline) — it removes the *avoidable*
    sources of run-to-run variance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_episode(
    env: SnakeEnv,
    agent: DQNAgent,
    episode_seed: Optional[int] = None,
) -> EpisodeStats:
    """Run a single training episode to completion and return its stats.

    ``terminated`` and ``truncated`` are kept distinct throughout (per
    Section 4 of the Phase 4 instructions) but both end the episode and
    both count as ``done=True`` for the replay buffer, matching the
    standard convention that a truncated episode still has no valid
    bootstrapped continuation within this same trajectory.
    """
    state, _info = env.reset(seed=episode_seed)
    terminated = False
    truncated = False

    total_reward = 0.0
    steps = 0
    losses: list[float] = []

    while not (terminated or truncated):
        action = agent.select_action(state, training=True)
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        agent.remember(state, action, reward, next_state, done)

        if agent.can_learn():
            loss = agent.learn()
            losses.append(loss)

        state = next_state
        total_reward += reward
        steps += 1

    agent.decay_epsilon()

    return EpisodeStats(
        episode=0,  # filled in by the caller, which tracks episode index
        reward=total_reward,
        score=info["score"],
        snake_length=info["snake_length"],
        steps=steps,
        epsilon=agent.epsilon,
        avg_loss=(sum(losses) / len(losses)) if losses else None,
        num_learning_updates=len(losses),
        termination_reason=info["termination_reason"],
    )


@dataclass
class TrainingResult:
    """Summary returned by :func:`train` for programmatic inspection
    (e.g. by tests or the smoke test), separate from what gets written
    to disk by the logger/checkpointer."""

    episodes_completed: int
    best_score: int
    best_model_path: Optional[Path]
    log_path: Path
    elapsed_seconds: float


def train(config: TrainingConfig, agent: Optional[DQNAgent] = None) -> TrainingResult:
    """Run the full training loop described by ``config``.

    Parameters
    ----------
    config:
        Training-loop settings (Section 8). DQN hyperparameters are not
        part of this object; construct/pass a pre-configured ``agent``
        if non-default DQN hyperparameters are needed.
    agent:
        An existing ``DQNAgent`` to continue training (e.g. after
        loading a checkpoint via ``config.resume_from``). If omitted, a
        new agent is constructed with default ``DQNAgent`` hyperparameters
        and seeded from ``config.seed``.
    """
    set_global_seed(config.seed)

    env = SnakeEnv(board_size=config.board_size, max_episode_steps=config.max_episode_steps)

    if agent is None:
        agent = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=int(env.action_space.n),
            seed=config.seed,
        )
    if config.resume_from is not None:
        agent.load(config.resume_from)

    models_dir = Path(config.models_dir)
    results_dir = Path(config.results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "training_log.csv"

    best_score = -1
    best_model_path: Optional[Path] = None
    start_time = time.time()

    with TrainingLogger(log_path) as logger:
        for episode in range(1, config.num_episodes + 1):
            # Per-episode seed derived from the master seed keeps each
            # episode's environment RNG reproducible while still varying
            # across episodes (a single fixed seed for every episode
            # would replay an identical food-spawn sequence forever).
            episode_seed = config.seed + episode
            stats = run_episode(env, agent, episode_seed=episode_seed)
            stats.episode = episode
            logger.log(stats)

            if stats.score > best_score:
                best_score = stats.score
                best_model_path = models_dir / "best_model.pth"
                agent.save(best_model_path)

            if config.checkpoint_interval > 0 and episode % config.checkpoint_interval == 0:
                checkpoint_path = models_dir / f"checkpoint_episode_{episode}.pth"
                agent.save(checkpoint_path)

            if config.log_interval > 0 and episode % config.log_interval == 0:
                loss_display = f"{stats.avg_loss:.4f}" if stats.avg_loss is not None else "n/a"
                print(
                    f"Episode {episode}/{config.num_episodes} | "
                    f"score={stats.score} reward={stats.reward:.2f} "
                    f"steps={stats.steps} epsilon={stats.epsilon:.3f} "
                    f"avg_loss={loss_display}"
                )

    env.close()

    return TrainingResult(
        episodes_completed=config.num_episodes,
        best_score=best_score,
        best_model_path=best_model_path,
        log_path=log_path,
        elapsed_seconds=time.time() - start_time,
    )
