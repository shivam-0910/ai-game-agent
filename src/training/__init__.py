# Source of truth:
# docs/phase_1_2_technical_specification.md
#
# Phase: 4 — DQN Training Pipeline

"""DQN training pipeline package.

Exposes the training loop, configuration, and logging primitives.
Evaluation, hyperparameter optimization, and reporting are out of
scope for this package (see later phases).
"""

from src.training.config import TrainingConfig
from src.training.logger import EpisodeStats, TrainingLogger
from src.training.train import TrainingResult, run_episode, set_global_seed, train

__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "EpisodeStats",
    "TrainingLogger",
    "train",
    "run_episode",
    "set_global_seed",
]
