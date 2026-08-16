"""Custom Gymnasium-compatible Snake environment.

Implements Phase 2 of the AI Game Agent project, following the design
decisions frozen in ``docs/phase_1_2_technical_specification.md``:

- 20x20 board (configurable).
- Relative action space: Straight / Turn Left / Turn Right (Section 4).
- Compact feature-based observation (Section 3, Approach A).
- Reward: food +10, death -10, step -0.01, distance shaping +-0.1
  (Section 5), with no survival bonus.
- Deterministic seeding via ``reset(seed=...)`` (Section 2, Section 11).
- Distinct ``terminated`` / ``truncated`` signals (Section 2).

Coordinate convention
----------------------
Positions are ``(row, col)`` pairs. Row 0 is the top of the board and
row increases downward; column 0 is the left edge and column increases
rightward. This matches standard array/image indexing conventions and
is used consistently across the grid, the snake body deque, and the
food position.

This module intentionally contains ONLY environment logic. No DQN,
network, replay buffer, or training code lives here (see Section 17 of
the Phase 2 instructions).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.environment.constants import (
    DEFAULT_BOARD_SIZE,
    DEFAULT_INITIAL_SNAKE_LENGTH,
    DIRECTION_DELTAS,
    EMPTY_CELL,
    FOOD_CELL,
    MAX_STEPS_PER_CELL_FACTOR,
    REWARD_AWAY_FROM_FOOD,
    REWARD_DEATH,
    REWARD_FOOD,
    REWARD_STEP,
    REWARD_TOWARD_FOOD,
    SNAKE_BODY_CELL,
    SNAKE_HEAD_CELL,
    Direction,
    RelativeAction,
)

# Number of features in the compact observation vector (Section 3 / 6).
# 3 danger flags (straight, left, right) + 4 heading one-hot
# + 4 food-direction flags (up, down, left, right relative to head) = 11.
OBSERVATION_SIZE = 11


class SnakeEnv(gym.Env):
    """A deterministic, Gymnasium-compatible Snake environment.

    Parameters
    ----------
    board_size:
        Side length of the square board (default 20, per Section 16).
    max_episode_steps:
        Optional explicit step cap. If ``None``, a default proportional
        to board size is used (``MAX_STEPS_PER_CELL_FACTOR * board_size**2``),
        per the open decision in Section 18 of the spec.
    render_mode:
        Either ``None`` or ``"ansi"``. Rendering is intentionally minimal
        (Section 13) and exists only to aid manual/visual debugging.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        board_size: int = DEFAULT_BOARD_SIZE,
        max_episode_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        if board_size < 5:
            raise ValueError("board_size must be at least 5 to fit a starting snake.")

        self.board_size = board_size
        self.max_episode_steps = (
            max_episode_steps
            if max_episode_steps is not None
            else MAX_STEPS_PER_CELL_FACTOR * (board_size ** 2)
        )
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")
        self.render_mode = render_mode

        # Gymnasium-required spaces.
        self.action_space = spaces.Discrete(len(RelativeAction))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )

        # RNG is created fresh on every reset(seed=...) call for
        # reproducibility; gym.Env provides self.np_random via seeding
        # helpers, but we manage it explicitly for clarity and control
        # over exactly what consumes randomness (food spawning only).
        self._rng: np.random.Generator = np.random.default_rng()

        # Episode state, all (re)initialized in reset().
        self.snake_body: deque[tuple[int, int]] = deque()
        self.direction: Direction = Direction.RIGHT
        self.food_pos: tuple[int, int] = (0, 0)
        self.steps_taken: int = 0
        self.score: int = 0
        self._prev_food_distance: int = 0
        self._terminated: bool = False
        self._truncated: bool = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to a fresh episode.

        Following Gymnasium convention, passing ``seed`` (re)seeds the
        environment's internal RNG so that food placement (and any other
        randomness) becomes reproducible for that and subsequent calls
        until reseeded again.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._place_snake()
        self.direction = Direction.RIGHT
        self.steps_taken = 0
        self.score = 0
        self._terminated = False
        self._truncated = False
        self._spawn_food()
        self._prev_food_distance = self._manhattan_distance_to_food(self.snake_body[0])

        observation = self._get_observation()
        info = self._get_info(termination_reason=None)
        return observation, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the environment by one step given a relative action."""
        if self._terminated or self._truncated:
            raise RuntimeError(
                "step() called on a finished episode; call reset() first."
            )
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action!r}")

        self.direction = self._apply_relative_action(self.direction, RelativeAction(action))

        head_row, head_col = self.snake_body[0]
        d_row, d_col = DIRECTION_DELTAS[self.direction]
        new_head = (head_row + d_row, head_col + d_col)

        self.steps_taken += 1

        reward = REWARD_STEP
        termination_reason: Optional[str] = None
        ate_food = new_head == self.food_pos

        if self._is_collision(new_head, moving_into_tail_ok=ate_food):
            self._terminated = True
            reward = REWARD_DEATH
            termination_reason = (
                "wall_collision" if not self._in_bounds(new_head) else "self_collision"
            )
            # Head does not actually move onto the board on a fatal
            # collision; body is left as-is for a stable terminal state.
        else:
            if ate_food:
                self.snake_body.appendleft(new_head)
                self.score += 1
                reward = REWARD_FOOD
                if len(self.snake_body) >= self.board_size * self.board_size:
                    self._terminated = True
                    termination_reason = "board_full"
                else:
                    self._spawn_food()
            else:
                self.snake_body.appendleft(new_head)
                self.snake_body.pop()

                new_distance = self._manhattan_distance_to_food(new_head)
                if new_distance < self._prev_food_distance:
                    reward += REWARD_TOWARD_FOOD
                elif new_distance > self._prev_food_distance:
                    reward += REWARD_AWAY_FROM_FOOD
                self._prev_food_distance = new_distance

        if not self._terminated and self.steps_taken >= self.max_episode_steps:
            self._truncated = True
            termination_reason = "step_limit"

        observation = self._get_observation()
        info = self._get_info(termination_reason=termination_reason)
        return observation, reward, self._terminated, self._truncated, info

    def render(self) -> Optional[str]:
        """Minimal text rendering for manual debugging (Section 13).

        A full graphical renderer is explicitly out of scope for this
        phase; an ANSI grid is sufficient to visually verify environment
        correctness during development.
        """
        if self.render_mode != "ansi":
            return None

        grid = [[EMPTY_CELL for _ in range(self.board_size)] for _ in range(self.board_size)]
        for row, col in self.snake_body:
            grid[row][col] = SNAKE_BODY_CELL
        head_row, head_col = self.snake_body[0]
        grid[head_row][head_col] = SNAKE_HEAD_CELL
        food_row, food_col = self.food_pos
        grid[food_row][food_col] = FOOD_CELL

        symbols = {EMPTY_CELL: ".", SNAKE_BODY_CELL: "o", SNAKE_HEAD_CELL: "H", FOOD_CELL: "F"}
        lines = ["".join(symbols[cell] for cell in row) for row in grid]
        return "\n".join(lines)

    def close(self) -> None:
        """No external resources are held; nothing to clean up."""
        return None

    # ------------------------------------------------------------------
    # Internal helpers: setup
    # ------------------------------------------------------------------

    def _place_snake(self) -> None:
        """Spawn the snake centered on the board, facing right (Section 2)."""
        center_row = self.board_size // 2
        center_col = self.board_size // 2
        length = min(DEFAULT_INITIAL_SNAKE_LENGTH, center_col + 1)

        # Head at the center; body trails to the left since the initial
        # heading is RIGHT. Head is index 0 (Section 2 spec table).
        self.snake_body = deque(
            (center_row, center_col - offset) for offset in range(length)
        )

    def _spawn_food(self) -> None:
        """Place food uniformly at random on an empty cell (Section 2).

        Re-rolls if the sampled cell lands on the snake body, guaranteeing
        a valid spawn. If the board has no empty cells at all, this is a
        win condition handled by the caller before ``_spawn_food`` would
        ever be invoked with zero free cells.
        """
        occupied = set(self.snake_body)
        free_cells = [
            (r, c)
            for r in range(self.board_size)
            for c in range(self.board_size)
            if (r, c) not in occupied
        ]
        if not free_cells:
            # Board is completely full; caller is responsible for ending
            # the episode as a "win" before reaching this state during
            # step(). Reaching here during reset() would only happen for
            # a pathologically tiny board, so fail loudly rather than
            # spin.
            raise RuntimeError("No free cells available to spawn food.")

        index = int(self._rng.integers(0, len(free_cells)))
        self.food_pos = free_cells[index]

    # ------------------------------------------------------------------
    # Internal helpers: movement / collision
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_relative_action(
        current_direction: Direction, action: RelativeAction
    ) -> Direction:
        """Rotate the current heading according to a relative action.

        Straight leaves heading unchanged; Turn Left rotates 90 degrees
        counter-clockwise; Turn Right rotates 90 degrees clockwise, per
        Section 4. The Direction enum is ordered UP -> RIGHT -> DOWN ->
        LEFT clockwise, so clockwise = +1 mod 4 and counter-clockwise =
        -1 mod 4.
        """
        if action == RelativeAction.STRAIGHT:
            return current_direction
        if action == RelativeAction.TURN_RIGHT:
            return Direction((current_direction + 1) % 4)
        # TURN_LEFT
        return Direction((current_direction - 1) % 4)

    def _in_bounds(self, pos: tuple[int, int]) -> bool:
        row, col = pos
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def _is_collision(self, new_head: tuple[int, int], moving_into_tail_ok: bool) -> bool:
        """Check wall or self collision for a candidate new head position.

        ``moving_into_tail_ok`` is unused for the tail-drop case because
        the tail only vacates its cell when the snake does NOT grow this
        step; when food is eaten the tail stays put, so self-collision
        must be checked against the full current body in that case too.
        The body used for the check already reflects this: when growing,
        the full current body (including the current tail cell) is the
        correct collision set, since the tail does not move away.
        """
        if not self._in_bounds(new_head):
            return True

        body_to_check = self.snake_body
        if not moving_into_tail_ok:
            # The tail cell will be vacated this step (no growth), so a
            # move onto the current tail position is legal.
            body_to_check = deque(list(self.snake_body)[:-1])

        return new_head in body_to_check

    # ------------------------------------------------------------------
    # Internal helpers: observation
    # ------------------------------------------------------------------

    def _manhattan_distance_to_food(self, pos: tuple[int, int]) -> int:
        row, col = pos
        food_row, food_col = self.food_pos
        return abs(row - food_row) + abs(col - food_col)

    def _get_observation(self) -> np.ndarray:
        """Build the compact feature vector (Section 3, Approach A).

        Layout (11 float32 values, each 0.0 or 1.0):

        Index | Meaning
        ------|--------------------------------------------------------
        0     | Danger straight ahead (moving Straight next step dies)
        1     | Danger to the left (moving Turn Left next step dies)
        2     | Danger to the right (moving Turn Right next step dies)
        3     | Heading is UP
        4     | Heading is RIGHT
        5     | Heading is DOWN
        6     | Heading is LEFT
        7     | Food is above the head (food_row < head_row)
        8     | Food is below the head (food_row > head_row)
        9     | Food is left of the head (food_col < head_col)
        10    | Food is right of the head (food_col > head_col)

        Danger flags are computed relative to current heading (matching
        the relative action space) rather than as absolute directions,
        so the feature meaning is heading-invariant from the agent's
        point of view. Food-direction flags are absolute (up/down/left/
        right in board coordinates) since they describe a spatial
        relationship the network can combine with the heading one-hot.
        """
        head_row, head_col = self.snake_body[0]

        straight_dir = self.direction
        left_dir = self._apply_relative_action(self.direction, RelativeAction.TURN_LEFT)
        right_dir = self._apply_relative_action(self.direction, RelativeAction.TURN_RIGHT)

        danger_straight = self._would_collide(head_row, head_col, straight_dir)
        danger_left = self._would_collide(head_row, head_col, left_dir)
        danger_right = self._would_collide(head_row, head_col, right_dir)

        heading_one_hot = [0.0, 0.0, 0.0, 0.0]
        heading_one_hot[int(self.direction)] = 1.0

        food_row, food_col = self.food_pos
        food_up = float(food_row < head_row)
        food_down = float(food_row > head_row)
        food_left = float(food_col < head_col)
        food_right = float(food_col > head_col)

        observation = np.array(
            [
                float(danger_straight),
                float(danger_left),
                float(danger_right),
                *heading_one_hot,
                food_up,
                food_down,
                food_left,
                food_right,
            ],
            dtype=np.float32,
        )
        return observation

    def _would_collide(self, head_row: int, head_col: int, direction: Direction) -> bool:
        """Whether moving one step in ``direction`` from the head would
        immediately hit a wall or the snake's own body (tail-aware, same
        rule as the real step: moving into the current tail cell is safe
        since the tail vacates unless food is eaten there, but we treat
        the tail cell conservatively as passable for the danger sensor).
        """
        d_row, d_col = DIRECTION_DELTAS[direction]
        candidate = (head_row + d_row, head_col + d_col)
        if not self._in_bounds(candidate):
            return True
        body_without_tail = deque(list(self.snake_body)[:-1])
        return candidate in body_without_tail

    # ------------------------------------------------------------------
    # Internal helpers: info
    # ------------------------------------------------------------------

    def _get_info(self, termination_reason: Optional[str]) -> dict[str, Any]:
        return {
            "score": self.score,
            "snake_length": len(self.snake_body),
            "food_position": self.food_pos,
            "steps": self.steps_taken,
            "termination_reason": termination_reason,
        }