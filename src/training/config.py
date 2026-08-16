# Source of truth:
# docs/phase_1_2_technical_specification.md
#
# Based on:
# src/agent/dqn_agent.py
# src/environment/snake_env.py
#
# Phase: 4 — DQN Training Pipeline

"""Training configuration for Phase 4.

Deliberately NOT a general configuration framework. This module owns
only the values that belong to the *training loop* (episode count,
seed, checkpoint/log intervals, directories) per Section 8 of the
Phase 4 instructions. DQN hyperparameters (learning rate, gamma,
epsilon schedule, replay buffer size, etc.) remain owned by
``DQNAgent`` and are not duplicated here — a ``TrainingConfig`` may
optionally *override* a couple of the most commonly-tuned ones when
constructing the agent, but it does not re-implement or shadow the
agent's own defaults.

A config can be built directly in Python (the common case for the
smoke test and tests) or loaded from a small JSON file under
``configs/`` for a named baseline run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    """Parameters owned by the training loop itself.

    Parameters
    ----------
    num_episodes:
        Total number of training episodes to run.
    seed:
        Master seed applied to Python ``random``, NumPy, PyTorch, and
        the environment's ``reset(seed=...)`` for reproducibility
        (Section 9 of the Phase 4 instructions).
    board_size:
        Passed straight through to ``SnakeEnv(board_size=...)``.
    max_episode_steps:
        Optional override passed to ``SnakeEnv``; ``None`` uses the
        environment's own default (proportional to board size).
    checkpoint_interval:
        Save a periodic checkpoint every N episodes. Set to 0 to
        disable periodic checkpoints (best-model checkpointing still
        applies).
    log_interval:
        Print a brief progress line to the console every N episodes.
        Does not affect what is written to the CSV log, which records
        every episode.
    models_dir / results_dir:
        Output directories for checkpoints and the training log,
        respectively. Created on demand if missing.
    resume_from:
        Optional path to a checkpoint to load via ``DQNAgent.load()``
        before training starts, continuing episode numbering from 0
        (episode numbering is a training-loop concern; the agent's own
        learned state, including epsilon, is what actually resumes).
    """

    num_episodes: int = 500
    seed: int = 42
    board_size: int = 20
    max_episode_steps: int | None = None
    checkpoint_interval: int = 100
    log_interval: int = 10
    models_dir: str = "models"
    results_dir: str = "results/training"
    resume_from: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        """Load a config from a JSON file under ``configs/``.

        Unknown keys in the file are rejected (rather than silently
        ignored) to catch typos early; missing keys fall back to the
        dataclass defaults above.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - valid_fields
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def save_json(self, path: str | Path) -> None:
        """Write this config to disk, e.g. alongside a checkpoint, so a
        run's exact settings are reproducible/inspectable later."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
