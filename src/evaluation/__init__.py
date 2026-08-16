# Source of truth:
# docs/phase_1_2_technical_specification.md
#
# Phase: 5 — DQN Evaluation

"""Evaluation pipeline package: greedy, no-learning evaluation of a
trained DQN checkpoint against SnakeEnv. No training or optimization
logic lives here (Section 14 of the Phase 5 instructions).
"""

from src.evaluation.evaluate import (
    EvaluationConfig,
    EvaluationRunOutput,
    EvaluationSummary,
    EpisodeResult,
    evaluate,
    run_evaluation,
)

__all__ = [
    "EvaluationConfig",
    "EvaluationRunOutput",
    "EvaluationSummary",
    "EpisodeResult",
    "evaluate",
    "run_evaluation",
]
