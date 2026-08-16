# Source of truth:
# docs/phase_1_2_technical_specification.md
#
# Based on:
# src/training/train.py
# src/training/config.py
#
# Phase: 4 — DQN Training Pipeline

"""Command-line entry point for a training run.

Usage:

    python -m src.training.run_training
    python -m src.training.run_training --config configs/baseline.json
    python -m src.training.run_training --episodes 20 --seed 0

This script only wires ``TrainingConfig`` to :func:`train` and prints a
short summary — no evaluation or reporting logic lives here.
"""

from __future__ import annotations

import argparse

from src.training.config import TrainingConfig
from src.training.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DQN Snake agent.")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to a JSON config under configs/."
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override num_episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = TrainingConfig.from_json(args.config) if args.config else TrainingConfig()
    if args.episodes is not None:
        config.num_episodes = args.episodes
    if args.seed is not None:
        config.seed = args.seed

    result = train(config)

    print("\nTraining complete.")
    print(f"  Episodes completed : {result.episodes_completed}")
    print(f"  Best score observed: {result.best_score}")
    print(f"  Best model saved to: {result.best_model_path}")
    print(f"  Training log        : {result.log_path}")
    print(f"  Elapsed time        : {result.elapsed_seconds:.1f}s")


if __name__ == "__main__":
    main()
