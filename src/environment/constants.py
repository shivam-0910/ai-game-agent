"""Constants, enums, and default configuration values for the Snake
environment.

Kept in a separate module (rather than inline magic numbers in
``snake_env.py``) purely for readability and single-source-of-truth
configuration, per the Phase 2 instructions on avoiding magic numbers.
This is not a general-purpose config framework.
"""

from __future__ import annotations

from enum import IntEnum


class Direction(IntEnum):
    """Absolute heading of the snake, used internally by the environment.

    Not exposed as the public action space (the public action space is
    relative — see :class:`RelativeAction`). Values are arbitrary but
    fixed so that turning logic (Section 4 of the spec) can be expressed
    as simple modular arithmetic over this ordering.

    The order UP -> RIGHT -> DOWN -> LEFT -> UP is the clockwise cycle
    used by ``turn_right`` / ``turn_left``.
    """

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class RelativeAction(IntEnum):
    """Public action space, relative to the snake's current heading.

    This is the *only* action interface the environment exposes to an
    agent. There is deliberately no absolute UP/DOWN/LEFT/RIGHT action,
    which makes an instant 180-degree reversal into the snake's own neck
    structurally unrepresentable (Section 4 of the technical spec).
    """

    STRAIGHT = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


# (row_delta, col_delta) for a single step in each absolute direction.
# Row 0 is the top of the board; row increases downward (standard
# array/image convention), col increases rightward.
DIRECTION_DELTAS: dict[Direction, tuple[int, int]] = {
    Direction.UP: (-1, 0),
    Direction.RIGHT: (0, 1),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
}

# Grid cell codes used by the internal 2D representation (Section 2 of
# the spec). Not part of the public observation (see Section 3 / Approach
# A) — used only for internal bookkeeping and optional rendering.
EMPTY_CELL = 0
SNAKE_BODY_CELL = 1
SNAKE_HEAD_CELL = 2
FOOD_CELL = 3

# --- Baseline configuration (Section 16 of the technical spec) ---

DEFAULT_BOARD_SIZE = 20
DEFAULT_INITIAL_SNAKE_LENGTH = 3

# Reward magnitudes (Section 5).
REWARD_FOOD = 10.0
REWARD_DEATH = -10.0
REWARD_STEP = -0.01
REWARD_TOWARD_FOOD = 0.1
REWARD_AWAY_FROM_FOOD = -0.1

# Episode step cap: "proportional to board size" per Section 2 / open
# decision in Section 18. A concrete formula is chosen here in Phase 2:
# cap = MAX_STEPS_PER_CELL_FACTOR * (board_size ** 2). This is generous
# enough not to cut off a competent policy while still forcibly ending
# any indefinite non-productive loop, per Section 2's termination rule.
MAX_STEPS_PER_CELL_FACTOR = 100
