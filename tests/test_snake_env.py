"""Test suite for the custom Snake environment (Phase 2).

Covers initialization, reset, movement, relative actions, food behavior,
collisions, rewards, episode termination/truncation, seeding/
reproducibility, observation validity, and Gymnasium compatibility, per
Section 13/14 of the technical specification.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from src.environment.constants import (
    REWARD_AWAY_FROM_FOOD,
    REWARD_DEATH,
    REWARD_FOOD,
    REWARD_STEP,
    REWARD_TOWARD_FOOD,
    Direction,
    RelativeAction,
)
from src.environment.snake_env import OBSERVATION_SIZE, SnakeEnv


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def env() -> SnakeEnv:
    return SnakeEnv(board_size=20)


@pytest.fixture
def small_env() -> SnakeEnv:
    """A smaller board makes some edge-case tests (wall collisions,
    full-board win condition) faster and simpler to set up."""
    return SnakeEnv(board_size=6)


# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------


class TestInitialization:
    def test_can_instantiate(self):
        env = SnakeEnv()
        assert env is not None

    def test_action_space_is_discrete_3(self, env: SnakeEnv):
        assert env.action_space.n == 3

    def test_observation_space_shape(self, env: SnakeEnv):
        assert env.observation_space.shape == (OBSERVATION_SIZE,)

    def test_observation_space_bounds(self, env: SnakeEnv):
        assert env.observation_space.low.min() == 0.0
        assert env.observation_space.high.max() == 1.0

    def test_default_board_size_is_20(self):
        env = SnakeEnv()
        assert env.board_size == 20

    def test_rejects_too_small_board(self):
        with pytest.raises(ValueError):
            SnakeEnv(board_size=2)


# ---------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------


class TestReset:
    def test_reset_returns_obs_and_info(self, env: SnakeEnv):
        obs, info = env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_observation_matches_space(self, env: SnakeEnv):
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)

    def test_reset_initial_snake_length(self, env: SnakeEnv):
        env.reset(seed=0)
        assert len(env.snake_body) == 3

    def test_reset_initial_snake_centered(self, env: SnakeEnv):
        env.reset(seed=0)
        head_row, head_col = env.snake_body[0]
        assert head_row == env.board_size // 2
        assert head_col == env.board_size // 2

    def test_reset_initial_direction_is_right(self, env: SnakeEnv):
        env.reset(seed=0)
        assert env.direction == Direction.RIGHT

    def test_reset_food_placed_on_board(self, env: SnakeEnv):
        env.reset(seed=0)
        row, col = env.food_pos
        assert 0 <= row < env.board_size
        assert 0 <= col < env.board_size

    def test_reset_food_not_on_snake(self, env: SnakeEnv):
        env.reset(seed=0)
        assert env.food_pos not in env.snake_body

    def test_reset_info_has_expected_keys(self, env: SnakeEnv):
        _, info = env.reset(seed=0)
        for key in ("score", "snake_length", "food_position", "steps", "termination_reason"):
            assert key in info

    def test_reset_score_and_steps_zeroed(self, env: SnakeEnv):
        env.reset(seed=0)
        assert env.score == 0
        assert env.steps_taken == 0


# ---------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------


class TestMovement:
    def test_straight_movement_moves_head_forward(self, env: SnakeEnv):
        env.reset(seed=0)
        head_before = env.snake_body[0]
        env.step(RelativeAction.STRAIGHT)
        head_after = env.snake_body[0]
        # Initial heading is RIGHT -> col increases by 1, row unchanged.
        assert head_after == (head_before[0], head_before[1] + 1)

    def test_left_turn_changes_heading(self, env: SnakeEnv):
        env.reset(seed=0)
        assert env.direction == Direction.RIGHT
        env.step(RelativeAction.TURN_LEFT)
        assert env.direction == Direction.UP

    def test_right_turn_changes_heading(self, env: SnakeEnv):
        env.reset(seed=0)
        assert env.direction == Direction.RIGHT
        env.step(RelativeAction.TURN_RIGHT)
        assert env.direction == Direction.DOWN

    def test_body_follows_head(self, env: SnakeEnv):
        env.reset(seed=0)
        body_before = list(env.snake_body)
        env.step(RelativeAction.STRAIGHT)
        body_after = list(env.snake_body)
        # No growth (no food eaten): length stays the same, and the old
        # head becomes the second segment.
        assert len(body_after) == len(body_before)
        assert body_after[1] == body_before[0]


# ---------------------------------------------------------------------
# Relative action mapping (Section 4) — all four headings
# ---------------------------------------------------------------------


class TestRelativeActionMapping:
    @pytest.mark.parametrize(
        "heading,expected_straight,expected_left,expected_right",
        [
            (Direction.UP, Direction.UP, Direction.LEFT, Direction.RIGHT),
            (Direction.RIGHT, Direction.RIGHT, Direction.UP, Direction.DOWN),
            (Direction.DOWN, Direction.DOWN, Direction.RIGHT, Direction.LEFT),
            (Direction.LEFT, Direction.LEFT, Direction.DOWN, Direction.UP),
        ],
    )
    def test_mapping_for_heading(
        self, heading, expected_straight, expected_left, expected_right
    ):
        assert SnakeEnv._apply_relative_action(heading, RelativeAction.STRAIGHT) == expected_straight
        assert SnakeEnv._apply_relative_action(heading, RelativeAction.TURN_LEFT) == expected_left
        assert SnakeEnv._apply_relative_action(heading, RelativeAction.TURN_RIGHT) == expected_right

    def test_reversal_is_unreachable_in_one_step(self):
        """No single relative action can produce the opposite heading."""
        opposite = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }
        for heading in Direction:
            reachable = {
                SnakeEnv._apply_relative_action(heading, a) for a in RelativeAction
            }
            assert opposite[heading] not in reachable


# ---------------------------------------------------------------------
# Food
# ---------------------------------------------------------------------


class TestFood:
    def test_food_inside_board_after_many_resets(self, env: SnakeEnv):
        for seed in range(20):
            env.reset(seed=seed)
            row, col = env.food_pos
            assert 0 <= row < env.board_size
            assert 0 <= col < env.board_size

    def test_food_never_spawns_on_snake_across_many_steps(self, env: SnakeEnv):
        env.reset(seed=1)
        for _ in range(200):
            if env._terminated or env._truncated:
                break
            assert env.food_pos not in list(env.snake_body)[1:] or env.food_pos == env.snake_body[0]
            env.step(RelativeAction.STRAIGHT)

    def test_eating_food_increases_score(self, env: SnakeEnv):
        env.reset(seed=0)
        _place_food_directly_ahead(env)
        _, reward, terminated, _, info = env.step(RelativeAction.STRAIGHT)
        assert info["score"] == 1
        assert reward == REWARD_FOOD
        assert not terminated

    def test_eating_food_grows_snake(self, env: SnakeEnv):
        env.reset(seed=0)
        length_before = len(env.snake_body)
        _place_food_directly_ahead(env)
        env.step(RelativeAction.STRAIGHT)
        assert len(env.snake_body) == length_before + 1

    def test_food_respawns_after_being_eaten(self, env: SnakeEnv):
        env.reset(seed=0)
        old_food = env.food_pos
        _place_food_directly_ahead(env)
        env.step(RelativeAction.STRAIGHT)
        assert env.food_pos != old_food
        assert env.food_pos not in env.snake_body


# ---------------------------------------------------------------------
# Collision
# ---------------------------------------------------------------------


class TestCollision:
    def test_wall_collision_terminates_episode(self, small_env: SnakeEnv):
        small_env.reset(seed=0)
        # Heading RIGHT from center; walk to the right wall.
        terminated = False
        info = {}
        for _ in range(small_env.board_size):
            _, _, terminated, _, info = small_env.step(RelativeAction.STRAIGHT)
            if terminated:
                break
        assert terminated
        assert info["termination_reason"] == "wall_collision"

    def test_wall_collision_reward_is_death_penalty(self, small_env: SnakeEnv):
        small_env.reset(seed=0)
        reward = None
        for _ in range(small_env.board_size):
            _, reward, terminated, _, _ = small_env.step(RelativeAction.STRAIGHT)
            if terminated:
                break
        assert reward == REWARD_DEATH

    def test_self_collision_terminates_episode(self, env: SnakeEnv):
        env.reset(seed=0)
        # Grow the snake long enough that a tight repeated-left-turn
        # spiral drives the head back into its own body within a few
        # steps (verified: length 11 self-collides on the 3rd turn).
        _grow_snake(env, times=8)

        terminated = False
        info = {}
        for _ in range(10):
            _, _, terminated, _, info = env.step(RelativeAction.TURN_LEFT)
            if terminated:
                break
        assert terminated
        assert info["termination_reason"] == "self_collision"


# ---------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------


class TestRewards:
    def test_step_penalty_when_nothing_happens(self, env: SnakeEnv):
        env.reset(seed=0)
        # Move straight without being near food/walls; on a 20x20 board
        # from center this is safe and won't eat immediately (food is
        # placed elsewhere with high probability; guard just in case).
        if env.food_pos == (
            env.snake_body[0][0],
            env.snake_body[0][1] + 1,
        ):
            env._spawn_food()  # nudge away from a coincidental adjacency for this test
        _, reward, _, _, _ = env.step(RelativeAction.STRAIGHT)
        assert reward in (
            REWARD_STEP + REWARD_TOWARD_FOOD,
            REWARD_STEP + REWARD_AWAY_FROM_FOOD,
            REWARD_FOOD,
        )

    def test_food_reward_value(self, env: SnakeEnv):
        env.reset(seed=0)
        _place_food_directly_ahead(env)
        _, reward, _, _, _ = env.step(RelativeAction.STRAIGHT)
        assert reward == REWARD_FOOD

    def test_death_reward_value(self, small_env: SnakeEnv):
        small_env.reset(seed=0)
        reward = None
        for _ in range(small_env.board_size):
            _, reward, terminated, _, _ = small_env.step(RelativeAction.STRAIGHT)
            if terminated:
                break
        assert reward == REWARD_DEATH

    def test_distance_shaping_toward_food(self, env: SnakeEnv):
        env.reset(seed=0)
        head_row, head_col = env.snake_body[0]
        env.food_pos = (head_row, head_col + 5)  # directly ahead, far away
        env._prev_food_distance = env._manhattan_distance_to_food((head_row, head_col))
        _, reward, _, _, _ = env.step(RelativeAction.STRAIGHT)
        assert reward == pytest.approx(REWARD_STEP + REWARD_TOWARD_FOOD)

    def test_distance_shaping_away_from_food(self, env: SnakeEnv):
        env.reset(seed=0)
        head_row, head_col = env.snake_body[0]
        env.food_pos = (head_row, head_col - 3)  # behind relative to RIGHT heading
        env._prev_food_distance = env._manhattan_distance_to_food((head_row, head_col))
        _, reward, _, _, _ = env.step(RelativeAction.STRAIGHT)
        assert reward == pytest.approx(REWARD_STEP + REWARD_AWAY_FROM_FOOD)

    def test_no_survival_bonus(self, env: SnakeEnv):
        """A pure non-eating, distance-neutral step must not be positive
        beyond the toward-food shaping term; there is no separate
        survival bonus term added on top."""
        env.reset(seed=0)
        head_row, head_col = env.snake_body[0]
        # Place food far to the side so moving straight (RIGHT) neither
        # strictly approaches nor recedes... not generally achievable
        # exactly, so instead assert the reward composition directly.
        env.food_pos = (head_row, head_col + 5)
        env._prev_food_distance = env._manhattan_distance_to_food((head_row, head_col))
        _, reward, _, _, _ = env.step(RelativeAction.STRAIGHT)
        # Reward must equal step penalty plus exactly one shaping term,
        # never step penalty alone with an extra positive bonus.
        assert reward == pytest.approx(REWARD_STEP + REWARD_TOWARD_FOOD)


# ---------------------------------------------------------------------
# Reward overrides (Phase 6)
# ---------------------------------------------------------------------


class TestRewardOverrides:
    """Tests for the optional reward_overrides constructor parameter
    added in Phase 6 (Section 12 of the spec: reward experiments)."""

    def test_default_matches_no_overrides(self):
        env_default = SnakeEnv(board_size=8)
        env_explicit_none = SnakeEnv(board_size=8, reward_overrides=None)
        assert env_default.reward_food == env_explicit_none.reward_food == REWARD_FOOD
        assert env_default.reward_death == env_explicit_none.reward_death == REWARD_DEATH
        assert env_default.reward_step == env_explicit_none.reward_step == REWARD_STEP

    def test_empty_dict_matches_defaults(self):
        env = SnakeEnv(board_size=8, reward_overrides={})
        assert env.reward_food == REWARD_FOOD
        assert env.reward_death == REWARD_DEATH

    def test_single_override_applied(self):
        env = SnakeEnv(board_size=8, reward_overrides={"REWARD_FOOD": 15.0})
        assert env.reward_food == 15.0
        # Everything else stays at the module default.
        assert env.reward_death == REWARD_DEATH
        assert env.reward_step == REWARD_STEP
        assert env.reward_toward_food == REWARD_TOWARD_FOOD
        assert env.reward_away_from_food == REWARD_AWAY_FROM_FOOD

    def test_multiple_overrides_applied(self):
        env = SnakeEnv(
            board_size=8, reward_overrides={"REWARD_FOOD": 20.0, "REWARD_DEATH": -5.0}
        )
        assert env.reward_food == 20.0
        assert env.reward_death == -5.0

    def test_unknown_override_key_raises(self):
        with pytest.raises(ValueError):
            SnakeEnv(board_size=8, reward_overrides={"NOT_A_REAL_KEY": 1.0})

    def test_override_actually_used_in_step(self):
        env = SnakeEnv(board_size=8, reward_overrides={"REWARD_FOOD": 42.0})
        env.reset(seed=0)
        _place_food_directly_ahead(env)
        _, reward, _, _, _ = env.step(RelativeAction.STRAIGHT)
        assert reward == 42.0

    def test_death_override_actually_used_in_step(self):
        env = SnakeEnv(board_size=8, reward_overrides={"REWARD_DEATH": -3.0})
        env.reset(seed=0)
        reward = None
        for _ in range(env.board_size):
            _, reward, terminated, _, _ = env.step(RelativeAction.STRAIGHT)
            if terminated:
                break
        assert reward == -3.0

    def test_module_constants_unaffected_by_instance_overrides(self):
        """Constructing an env with overrides must not mutate the
        shared module-level constants used by other SnakeEnv instances."""
        SnakeEnv(board_size=8, reward_overrides={"REWARD_FOOD": 999.0})
        fresh_env = SnakeEnv(board_size=8)
        assert fresh_env.reward_food == REWARD_FOOD
        assert REWARD_FOOD == 10.0  # the original module constant itself


# ---------------------------------------------------------------------
# Episode handling
# ---------------------------------------------------------------------


class TestEpisodeHandling:
    def test_truncated_on_step_limit(self):
        env = SnakeEnv(board_size=20, max_episode_steps=5)
        env.reset(seed=0)
        truncated = False
        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(RelativeAction.TURN_LEFT)
            env.step(RelativeAction.TURN_RIGHT) if False else None
            if terminated or truncated:
                break
        assert truncated

    def test_step_raises_after_termination(self, small_env: SnakeEnv):
        small_env.reset(seed=0)
        terminated = False
        for _ in range(small_env.board_size):
            _, _, terminated, _, _ = small_env.step(RelativeAction.STRAIGHT)
            if terminated:
                break
        assert terminated
        with pytest.raises(RuntimeError):
            small_env.step(RelativeAction.STRAIGHT)

    def test_terminated_and_truncated_mutually_distinct_on_wall_death(self, small_env: SnakeEnv):
        small_env.reset(seed=0)
        terminated = truncated = False
        for _ in range(small_env.board_size):
            _, _, terminated, truncated, _ = small_env.step(RelativeAction.STRAIGHT)
            if terminated or truncated:
                break
        assert terminated is True
        assert truncated is False


# ---------------------------------------------------------------------
# Seeding / reproducibility
# ---------------------------------------------------------------------


class TestSeeding:
    def test_same_seed_same_initial_food(self, env: SnakeEnv):
        _, info_a = env.reset(seed=42)
        food_a = env.food_pos
        _, info_b = env.reset(seed=42)
        food_b = env.food_pos
        assert food_a == food_b

    def test_same_seed_same_trajectory(self):
        env_a = SnakeEnv(board_size=20)
        env_b = SnakeEnv(board_size=20)
        env_a.reset(seed=7)
        env_b.reset(seed=7)

        actions = [
            RelativeAction.STRAIGHT,
            RelativeAction.TURN_LEFT,
            RelativeAction.STRAIGHT,
            RelativeAction.TURN_RIGHT,
            RelativeAction.STRAIGHT,
        ]
        for action in actions:
            obs_a, reward_a, term_a, trunc_a, _ = env_a.step(action)
            obs_b, reward_b, term_b, trunc_b, _ = env_b.step(action)
            np.testing.assert_array_equal(obs_a, obs_b)
            assert reward_a == reward_b
            assert term_a == term_b
            assert trunc_a == trunc_b
            if term_a or trunc_a:
                break

    def test_different_seeds_can_differ(self):
        """Not a strict guarantee for every pair, but across many seed
        pairs on a 20x20 board at least one difference should appear,
        confirming the seed actually drives food placement."""
        positions = set()
        for seed in range(10):
            env = SnakeEnv(board_size=20)
            env.reset(seed=seed)
            positions.add(env.food_pos)
        assert len(positions) > 1


# ---------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------


class TestObservation:
    def test_observation_shape(self, env: SnakeEnv):
        obs, _ = env.reset(seed=0)
        assert obs.shape == (OBSERVATION_SIZE,)

    def test_observation_dtype(self, env: SnakeEnv):
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32

    def test_observation_within_bounds_over_episode(self, env: SnakeEnv):
        obs, _ = env.reset(seed=3)
        assert env.observation_space.contains(obs)
        for _ in range(100):
            if env._terminated or env._truncated:
                break
            obs, _, terminated, truncated, _ = env.step(RelativeAction.STRAIGHT)
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break

    def test_heading_one_hot_sums_to_one(self, env: SnakeEnv):
        obs, _ = env.reset(seed=0)
        heading_slice = obs[3:7]
        assert heading_slice.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------
# Gymnasium compatibility
# ---------------------------------------------------------------------


class TestGymnasiumCompatibility:
    def test_check_env_passes(self):
        from gymnasium.utils.env_checker import check_env

        env = SnakeEnv(board_size=20)
        # skip_render_check=True: our minimal "ansi" renderer is a debug
        # aid, not a fully spec-compliant Gymnasium render pipeline.
        check_env(env.unwrapped, skip_render_check=True)

    def test_can_execute_many_steps_without_error(self, env: SnakeEnv):
        env.reset(seed=0)
        rng = np.random.default_rng(123)
        for _ in range(300):
            if env._terminated or env._truncated:
                env.reset()
            action = int(rng.integers(0, 3))
            env.step(action)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _place_food_directly_ahead(env: SnakeEnv) -> None:
    """Force the food onto the cell the snake will move into next,
    given its current heading, for deterministic food-eating tests."""
    head_row, head_col = env.snake_body[0]
    d_row, d_col = {
        Direction.UP: (-1, 0),
        Direction.RIGHT: (0, 1),
        Direction.DOWN: (1, 0),
        Direction.LEFT: (0, -1),
    }[env.direction]
    env.food_pos = (head_row + d_row, head_col + d_col)


def _grow_snake(env: SnakeEnv, times: int) -> None:
    """Grow the snake by repeatedly placing food directly ahead and
    eating it, without changing heading."""
    for _ in range(times):
        _place_food_directly_ahead(env)
        env.step(RelativeAction.STRAIGHT)