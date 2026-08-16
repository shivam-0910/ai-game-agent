# Source of truth:
# docs/phase_1_2_technical_specification.md
#
# Based on:
# src/evaluation/evaluate.py
#
# Phase: 5 — DQN Evaluation

"""Command-line entry point for evaluating a trained checkpoint.

Usage:

    python -m src.evaluation.run_evaluation
    python -m src.evaluation.run_evaluation --checkpoint models/best_model.pth --episodes 100
    python -m src.evaluation.run_evaluation --episodes 20 --seed 0 --verbose

This script only wires ``EvaluationConfig`` to :func:`evaluate` and
prints a short summary — it does not train, retrain, or modify the
checkpoint in any way.
"""

from __future__ import annotations

import argparse

from src.evaluation.evaluate import EvaluationConfig, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN Snake checkpoint.")
    parser.add_argument(
        "--checkpoint", type=str, default="models/best_model.pth", help="Path to the checkpoint."
    )
    parser.add_argument("--episodes", type=int, default=None, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Base evaluation seed.")
    parser.add_argument("--board-size", type=int, default=None, help="Override board size.")
    parser.add_argument(
        "--results-dir", type=str, default=None, help="Directory for evaluation output files."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print a line per evaluation episode."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = EvaluationConfig(checkpoint_path=args.checkpoint)
    if args.episodes is not None:
        config.num_episodes = args.episodes
    if args.seed is not None:
        config.base_seed = args.seed
    if args.board_size is not None:
        config.board_size = args.board_size
    if args.results_dir is not None:
        config.results_dir = args.results_dir
    if args.verbose:
        config.verbose = True

    output = evaluate(config)
    summary = output.summary

    print("\nEvaluation complete.")
    print(f"  Checkpoint          : {summary.checkpoint_path}")
    print(f"  Episodes            : {summary.num_episodes}")
    print(f"  Average score       : {summary.average_score:.2f}")
    print(f"  Max score           : {summary.max_score}")
    print(f"  Min score           : {summary.min_score}")
    print(f"  Score std. dev.     : {summary.score_std_dev:.2f}")
    print(f"  Average reward      : {summary.average_reward:.2f}")
    print(f"  Average steps       : {summary.average_steps:.2f}")
    print(f"  Average snake length: {summary.average_snake_length:.2f}")
    print(f"  Termination counts  : {summary.termination_counts}")
    print(f"  Results CSV         : {output.results_csv_path}")
    print(f"  Summary JSON        : {output.summary_json_path}")


if __name__ == "__main__":
    main()
