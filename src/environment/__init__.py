"""Custom Snake game environment package.

Exposes the Gymnasium-compatible SnakeEnv for use by future agent code
(Phase 3+). No agent, network, or training logic lives in this package.
"""

from src.environment.snake_env import SnakeEnv

__all__ = ["SnakeEnv"]
