# Source of truth:
# docs/phase_1_2_technical_specification.md
#
# Based on:
# src/evaluation/evaluate.py
# src/agent/dqn_agent.py
# src/environment/snake_env.py
#
# Phase: 5 — DQN Evaluation

"""Test suite for the evaluation pipeline (Phase 5).

All checkpoints used by these tests are freshly saved under
``tmp_path`` via ``DQNAgent.save()`` (Section 17 of the Phase 5
instructions) — the real ``models/best_model.pth`` is never required
by, or used in, the automated test suite. Boards are kept small and
episode counts low so the suite stays fast.
"""

from __future__ import annotations

import csv
import json

import pytest
import torch

from src.agent.dqn_agent import DQNAgent
from src.environment.snake_env import SnakeEnv
from src.evaluation.evaluate import (
    EvaluationConfig,
    evaluate,
    load_agent_for_evaluation,
    run_evaluation,
)


def _tiny_env() -> SnakeEnv:
    return SnakeEnv(board_size=8, max_episode_steps=100)


def _fresh_checkpoint(tmp_path, seed: int = 0) -> str:
    """Save a freshly-initialized (untrained, but valid) checkpoint via
    the real DQNAgent.save() API, for tests that just need *a*
    loadable checkpoint rather than a trained one."""
    env = _tiny_env()
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=int(env.action_space.n),
        seed=seed,
    )
    path = tmp_path / "checkpoint.pth"
    agent.save(path)
    return str(path)


def _tiny_config(tmp_path, checkpoint_path: str, **overrides) -> EvaluationConfig:
    defaults = dict(
        checkpoint_path=checkpoint_path,
        num_episodes=3,
        base_seed=999_000,
        board_size=8,
        max_episode_steps=100,
        results_dir=str(tmp_path / "results" / "evaluation"),
    )
    defaults.update(overrides)
    return EvaluationConfig(**defaults)


# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------


class TestInitialization:
    def test_config_constructs_with_defaults(self):
        config = EvaluationConfig()
        assert config.num_episodes > 0
        assert config.checkpoint_path == "models/best_model.pth"

    def test_config_accepts_overrides(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=5, base_seed=42)
        assert config.num_episodes == 5
        assert config.base_seed == 42
        assert config.checkpoint_path == checkpoint_path


# ---------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------


class TestCheckpointLoading:
    def test_valid_checkpoint_loads(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        env = _tiny_env()
        agent = load_agent_for_evaluation(checkpoint_path, env)
        assert isinstance(agent, DQNAgent)

    def test_missing_checkpoint_raises_clear_error(self, tmp_path):
        env = _tiny_env()
        missing_path = tmp_path / "does_not_exist.pth"
        with pytest.raises(FileNotFoundError):
            load_agent_for_evaluation(missing_path, env)

    def test_run_evaluation_raises_for_missing_checkpoint(self, tmp_path):
        config = _tiny_config(tmp_path, str(tmp_path / "missing.pth"))
        with pytest.raises(FileNotFoundError):
            run_evaluation(config)


# ---------------------------------------------------------------------
# Greedy behavior / no exploration
# ---------------------------------------------------------------------


class TestGreedyBehavior:
    def test_repeated_calls_same_state_produce_same_action(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        env = _tiny_env()
        agent = load_agent_for_evaluation(checkpoint_path, env)
        state, _info = env.reset(seed=1)
        actions = {agent.select_action(state, training=False) for _ in range(20)}
        assert len(actions) == 1

    def test_evaluation_episodes_use_greedy_selection_not_random(self, tmp_path, monkeypatch):
        """If evaluation ever used training=True (epsilon-greedy) instead
        of the greedy path, DQNAgent.select_action would consult
        random.random(). We assert it never does during evaluation by
        making random.random() explode and confirming evaluation still
        completes."""
        import random

        checkpoint_path = _fresh_checkpoint(tmp_path)
        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=2)

        def _boom(*args, **kwargs):
            raise AssertionError("random.random() must not be called during greedy evaluation")

        monkeypatch.setattr(random, "random", _boom)
        # Must not raise.
        episode_results, summary = run_evaluation(config)
        assert summary.num_episodes == 2


# ---------------------------------------------------------------------
# No learning / model integrity
# ---------------------------------------------------------------------


class TestNoLearning:
    def test_learn_is_never_called(self, tmp_path, monkeypatch):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=2)

        env = _tiny_env()
        agent = load_agent_for_evaluation(checkpoint_path, env)

        def _boom():
            raise AssertionError("agent.learn() must not be called during evaluation")

        monkeypatch.setattr(agent, "learn", _boom)
        # Must not raise.
        run_evaluation(config, agent=agent)

    def test_model_parameters_unchanged_after_evaluation(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        env = _tiny_env()
        agent = load_agent_for_evaluation(checkpoint_path, env)

        before = [p.clone() for p in agent.policy_network.parameters()]

        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=3)
        run_evaluation(config, agent=agent)

        after = list(agent.policy_network.parameters())
        for p_before, p_after in zip(before, after):
            assert torch.equal(p_before, p_after)

    def test_epsilon_not_decayed_during_evaluation(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        env = _tiny_env()
        agent = load_agent_for_evaluation(checkpoint_path, env)
        epsilon_before = agent.epsilon

        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=3)
        run_evaluation(config, agent=agent)

        assert agent.epsilon == epsilon_before

    def test_checkpoint_file_left_unmodified(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        original_bytes = (tmp_path / "checkpoint.pth").read_bytes()

        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=3)
        run_evaluation(config)

        assert (tmp_path / "checkpoint.pth").read_bytes() == original_bytes


# ---------------------------------------------------------------------
# No replay modification
# ---------------------------------------------------------------------


class TestNoReplayModification:
    def test_replay_buffer_not_populated(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        env = _tiny_env()
        agent = load_agent_for_evaluation(checkpoint_path, env)
        assert len(agent.replay_buffer) == 0

        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=3)
        run_evaluation(config, agent=agent)

        assert len(agent.replay_buffer) == 0

    def test_remember_never_called(self, tmp_path, monkeypatch):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        env = _tiny_env()
        agent = load_agent_for_evaluation(checkpoint_path, env)

        def _boom(*args, **kwargs):
            raise AssertionError("agent.remember() must not be called during evaluation")

        monkeypatch.setattr(agent, "remember", _boom)
        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=2)
        run_evaluation(config, agent=agent)


# ---------------------------------------------------------------------
# Episode execution
# ---------------------------------------------------------------------


class TestEpisodeExecution:
    def test_small_evaluation_run_completes(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=3)
        episode_results, summary = run_evaluation(config)
        assert len(episode_results) == 3
        assert summary.num_episodes == 3

    def test_episode_numbers_are_sequential_from_one(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=4)
        episode_results, _summary = run_evaluation(config)
        assert [r.episode for r in episode_results] == [1, 2, 3, 4]


# ---------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------


class TestSeeding:
    def test_each_episode_gets_distinct_seed(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=5, base_seed=100)
        episode_results, _summary = run_evaluation(config)
        seeds = [r.seed for r in episode_results]
        assert seeds == [100, 101, 102, 103, 104]
        assert len(set(seeds)) == len(seeds)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


class TestMetrics:
    def test_per_episode_results_have_expected_fields(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=2)
        episode_results, _summary = run_evaluation(config)
        for r in episode_results:
            assert isinstance(r.episode, int)
            assert isinstance(r.score, int)
            assert isinstance(r.reward, float)
            assert isinstance(r.steps, int)
            assert isinstance(r.snake_length, int)
            assert isinstance(r.seed, int)
            assert r.termination_reason in {
                "wall_collision",
                "self_collision",
                "board_full",
                "step_limit",
            }


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------


class TestAggregation:
    def test_average_score_matches_manual_calculation(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=5)
        episode_results, summary = run_evaluation(config)
        expected_avg = sum(r.score for r in episode_results) / len(episode_results)
        assert summary.average_score == pytest.approx(expected_avg)

    def test_max_and_min_score_correct(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=5)
        episode_results, summary = run_evaluation(config)
        scores = [r.score for r in episode_results]
        assert summary.max_score == max(scores)
        assert summary.min_score == min(scores)

    def test_termination_counts_sum_to_episode_count(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=6)
        _episode_results, summary = run_evaluation(config)
        assert sum(summary.termination_counts.values()) == 6

    def test_termination_percentages_sum_to_100(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=6)
        _episode_results, summary = run_evaluation(config)
        assert sum(summary.termination_percentages.values()) == pytest.approx(100.0)

    def test_single_episode_std_dev_is_zero(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=1)
        _episode_results, summary = run_evaluation(config)
        assert summary.score_std_dev == 0.0


# ---------------------------------------------------------------------
# CSV / JSON output
# ---------------------------------------------------------------------


class TestOutputFiles:
    def test_csv_created_with_expected_records(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=4)
        output = evaluate(config)
        assert output.results_csv_path.exists()
        with open(output.results_csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4
        assert set(rows[0].keys()) == {
            "episode",
            "score",
            "reward",
            "steps",
            "snake_length",
            "termination_reason",
            "seed",
        }

    def test_json_summary_created_and_valid(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=3)
        output = evaluate(config)
        assert output.summary_json_path.exists()
        with open(output.summary_json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["num_episodes"] == 3
        assert "average_score" in data
        assert "score_std_dev" in data
        assert "termination_counts" in data

    def test_does_not_overwrite_training_log(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path)
        training_dir = tmp_path / "results" / "training"
        training_dir.mkdir(parents=True)
        training_log = training_dir / "training_log.csv"
        training_log.write_text("episode,score\n1,0\n")

        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=2)
        evaluate(config)

        # Evaluation must write only under results/evaluation, never
        # touching results/training/training_log.csv.
        assert training_log.read_text() == "episode,score\n1,0\n"


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


class TestReproducibility:
    def test_same_checkpoint_and_seed_produce_same_results(self, tmp_path):
        checkpoint_path = _fresh_checkpoint(tmp_path, seed=5)
        config = _tiny_config(tmp_path, checkpoint_path, num_episodes=4, base_seed=777)

        results_a, summary_a = run_evaluation(config)
        results_b, summary_b = run_evaluation(config)

        assert [r.score for r in results_a] == [r.score for r in results_b]
        assert [r.steps for r in results_a] == [r.steps for r in results_b]
        assert summary_a.average_score == summary_b.average_score
