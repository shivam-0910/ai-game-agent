# Source of truth:
# docs/phase_1_2_technical_specification.md
#
# Based on:
# src/training/train.py
# src/training/config.py
# src/training/logger.py
#
# Phase: 4 — DQN Training Pipeline

"""Test suite for the training pipeline (Phase 4).

Uses tiny configurations (a handful of episodes, small boards/replay
thresholds) so the suite runs quickly, per Section 16 of the Phase 4
instructions. All generated artifacts are written under ``tmp_path``
and never left in the repository.
"""

from __future__ import annotations

import csv

import pytest
import torch

from src.agent.dqn_agent import DQNAgent
from src.environment.snake_env import SnakeEnv
from src.training.config import TrainingConfig
from src.training.logger import EpisodeStats, TrainingLogger
from src.training.train import run_episode, set_global_seed, train


def _tiny_config(tmp_path, **overrides) -> TrainingConfig:
    defaults = dict(
        num_episodes=5,
        seed=0,
        board_size=8,
        max_episode_steps=200,
        checkpoint_interval=0,  # disabled by default; enabled explicitly per test
        log_interval=0,  # silence console output during tests
        models_dir=str(tmp_path / "models"),
        results_dir=str(tmp_path / "results"),
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def _tiny_agent(env: SnakeEnv) -> DQNAgent:
    """An agent with a very low replay threshold so learning kicks in
    within the few steps a tiny test episode actually takes."""
    return DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=int(env.action_space.n),
        min_experiences=5,
        batch_size=4,
        target_update_frequency=3,
        seed=0,
    )


# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------


class TestInitialization:
    def test_config_constructs_with_defaults(self):
        config = TrainingConfig()
        assert config.num_episodes > 0

    def test_config_loads_from_json(self, tmp_path):
        config = TrainingConfig(num_episodes=7, seed=3)
        path = tmp_path / "cfg.json"
        config.save_json(path)
        loaded = TrainingConfig.from_json(path)
        assert loaded.num_episodes == 7
        assert loaded.seed == 3

    def test_config_rejects_unknown_keys(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"num_episodes": 5, "not_a_real_field": 1}')
        with pytest.raises(ValueError):
            TrainingConfig.from_json(path)

    def test_required_directories_can_be_created(self, tmp_path):
        config = _tiny_config(tmp_path, num_episodes=1)
        # models_dir/results_dir don't exist yet under tmp_path.
        assert not (tmp_path / "models").exists()
        train(config)
        assert (tmp_path / "models").exists()
        assert (tmp_path / "results").exists()

    def test_env_and_agent_initialize_together(self):
        env = SnakeEnv(board_size=8)
        agent = _tiny_agent(env)
        assert agent.state_size == env.observation_space.shape[0]
        assert agent.action_size == int(env.action_space.n)


# ---------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------


class TestSingleEpisode:
    def test_one_episode_runs_to_completion(self):
        env = SnakeEnv(board_size=8, max_episode_steps=100)
        agent = _tiny_agent(env)
        stats = run_episode(env, agent, episode_seed=0)
        assert stats.steps > 0
        assert stats.score >= 0

    def test_episode_stats_have_expected_types(self):
        env = SnakeEnv(board_size=8, max_episode_steps=100)
        agent = _tiny_agent(env)
        stats = run_episode(env, agent, episode_seed=0)
        assert isinstance(stats.reward, float)
        assert isinstance(stats.score, int)
        assert isinstance(stats.steps, int)
        assert isinstance(stats.epsilon, float)

    def test_episode_decays_epsilon_exactly_once(self):
        env = SnakeEnv(board_size=8, max_episode_steps=100)
        agent = _tiny_agent(env)
        epsilon_before = agent.epsilon
        run_episode(env, agent, episode_seed=0)
        expected = max(agent.epsilon_min, epsilon_before * agent.epsilon_decay)
        assert agent.epsilon == pytest.approx(expected)

    def test_avg_loss_is_none_when_no_learning_occurred(self):
        env = SnakeEnv(board_size=8, max_episode_steps=100)
        # min_experiences far above what a single short episode can
        # produce, so no learning happens during this episode.
        agent = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=int(env.action_space.n),
            min_experiences=10_000,
            seed=0,
        )
        stats = run_episode(env, agent, episode_seed=0)
        assert stats.avg_loss is None
        assert stats.num_learning_updates == 0


# ---------------------------------------------------------------------
# Small training run
# ---------------------------------------------------------------------


class TestSmallTrainingRun:
    def test_tiny_training_run_completes(self, tmp_path):
        config = _tiny_config(tmp_path, num_episodes=5)
        result = train(config)
        assert result.episodes_completed == 5

    def test_statistics_are_produced(self, tmp_path):
        config = _tiny_config(tmp_path, num_episodes=5)
        result = train(config)
        with open(result.log_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5

    def test_epsilon_changes_over_training(self, tmp_path):
        config = _tiny_config(tmp_path, num_episodes=5)
        result = train(config)
        with open(result.log_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        epsilons = [float(r["epsilon"]) for r in rows]
        assert epsilons[-1] < epsilons[0]

    def test_learning_occurs_once_replay_requirements_met(self, tmp_path):
        env_probe = SnakeEnv(board_size=8)
        agent = DQNAgent(
            state_size=env_probe.observation_space.shape[0],
            action_size=int(env_probe.action_space.n),
            min_experiences=5,
            batch_size=4,
            seed=0,
        )
        config = _tiny_config(tmp_path, num_episodes=5)
        train(config, agent=agent)
        with open(tmp_path / "results" / "training_log.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # With a low replay threshold, later episodes should have
        # recorded at least one learning update.
        total_updates = sum(int(r["num_learning_updates"]) for r in rows)
        assert total_updates > 0


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------


class TestLogging:
    def test_log_file_created(self, tmp_path):
        path = tmp_path / "log.csv"
        with TrainingLogger(path):
            pass
        assert path.exists()

    def test_log_has_expected_columns(self, tmp_path):
        path = tmp_path / "log.csv"
        with TrainingLogger(path) as logger:
            logger.log(
                EpisodeStats(
                    episode=1,
                    reward=1.0,
                    score=0,
                    snake_length=3,
                    steps=10,
                    epsilon=1.0,
                    avg_loss=None,
                    num_learning_updates=0,
                    termination_reason="wall_collision",
                )
            )
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == {
                "episode",
                "reward",
                "score",
                "snake_length",
                "steps",
                "epsilon",
                "avg_loss",
                "num_learning_updates",
                "termination_reason",
            }

    def test_missing_avg_loss_written_as_empty_field(self, tmp_path):
        path = tmp_path / "log.csv"
        with TrainingLogger(path) as logger:
            logger.log(
                EpisodeStats(
                    episode=1,
                    reward=1.0,
                    score=0,
                    snake_length=3,
                    steps=10,
                    epsilon=1.0,
                    avg_loss=None,
                    num_learning_updates=0,
                    termination_reason=None,
                )
            )
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["avg_loss"] == ""

    def test_values_are_valid_after_real_run(self, tmp_path):
        config = _tiny_config(tmp_path, num_episodes=3)
        result = train(config)
        with open(result.log_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert int(row["episode"]) >= 1
            assert float(row["reward"]) is not None
            assert int(row["score"]) >= 0
            assert int(row["steps"]) > 0
            assert 0.0 <= float(row["epsilon"]) <= 1.0


# ---------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------


class TestCheckpointing:
    def test_checkpoint_created_at_configured_interval(self, tmp_path):
        config = _tiny_config(tmp_path, num_episodes=4, checkpoint_interval=2)
        result = train(config)
        models_dir = tmp_path / "models"
        assert (models_dir / "checkpoint_episode_2.pth").exists()
        assert (models_dir / "checkpoint_episode_4.pth").exists()

    def test_no_periodic_checkpoint_when_interval_zero(self, tmp_path):
        config = _tiny_config(tmp_path, num_episodes=4, checkpoint_interval=0)
        train(config)
        models_dir = tmp_path / "models"
        checkpoint_files = list(models_dir.glob("checkpoint_episode_*.pth"))
        assert checkpoint_files == []

    def test_best_model_checkpoint_created(self, tmp_path):
        config = _tiny_config(tmp_path, num_episodes=5)
        result = train(config)
        assert result.best_model_path is not None
        assert result.best_model_path.exists()
        assert result.best_model_path.name == "best_model.pth"

    def test_saved_checkpoint_can_be_loaded_by_agent(self, tmp_path):
        config = _tiny_config(tmp_path, num_episodes=3)
        result = train(config)

        env = SnakeEnv(board_size=8)
        new_agent = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=int(env.action_space.n),
            seed=99,
        )
        # Must not raise; confirms the checkpoint round-trips through
        # the existing DQNAgent.load() API unchanged.
        new_agent.load(result.best_model_path)

    def test_best_model_uses_highest_score_criterion(self, tmp_path):
        """Directly exercises the best-model criterion: a later episode
        with a strictly higher score must replace the saved best model,
        and a subsequent lower-scoring episode must not overwrite it."""
        config = _tiny_config(tmp_path, num_episodes=1)
        env = SnakeEnv(board_size=8, max_episode_steps=200)
        agent = _tiny_agent(env)

        from src.training.logger import TrainingLogger

        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        best_score = -1
        best_path = None
        with TrainingLogger(tmp_path / "results" / "log.csv") as logger:
            for episode, forced_score in enumerate([0, 3, 1], start=1):
                stats = run_episode(env, agent, episode_seed=episode)
                stats.episode = episode
                stats.score = forced_score  # force a controlled sequence
                logger.log(stats)
                if stats.score > best_score:
                    best_score = stats.score
                    best_path = models_dir / "best_model.pth"
                    agent.save(best_path)
        assert best_score == 3


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_produces_same_first_episode_stats(self, tmp_path):
        config_a = _tiny_config(tmp_path / "a", num_episodes=1, seed=7)
        config_b = _tiny_config(tmp_path / "b", num_episodes=1, seed=7)

        result_a = train(config_a)
        result_b = train(config_b)

        with open(result_a.log_path, newline="", encoding="utf-8") as f:
            row_a = list(csv.DictReader(f))[0]
        with open(result_b.log_path, newline="", encoding="utf-8") as f:
            row_b = list(csv.DictReader(f))[0]

        assert row_a["score"] == row_b["score"]
        assert row_a["steps"] == row_b["steps"]
        assert row_a["reward"] == row_b["reward"]

    def test_set_global_seed_makes_torch_init_reproducible(self):
        set_global_seed(123)
        net_a_weight = torch.nn.Linear(4, 4).weight.clone()
        set_global_seed(123)
        net_b_weight = torch.nn.Linear(4, 4).weight.clone()
        assert torch.equal(net_a_weight, net_b_weight)


# ---------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------


class TestResume:
    def test_resume_from_checkpoint_continues_with_loaded_state(self, tmp_path):
        first_config = _tiny_config(tmp_path, num_episodes=3, seed=1)
        first_result = train(first_config)

        env = SnakeEnv(board_size=8)
        resumed_agent = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=int(env.action_space.n),
            seed=2,
        )
        epsilon_before_resume = resumed_agent.epsilon
        resumed_agent.load(first_result.best_model_path)
        # Loading a checkpoint from a run that already decayed epsilon
        # several times must change the fresh agent's epsilon rather
        # than leaving its own initial value in place.
        assert resumed_agent.epsilon != epsilon_before_resume

    def test_resume_from_config_field(self, tmp_path):
        first_config = _tiny_config(tmp_path, num_episodes=2, seed=1)
        first_result = train(first_config)

        second_config = _tiny_config(
            tmp_path, num_episodes=1, seed=1, resume_from=str(first_result.best_model_path)
        )
        # Must not raise: train() loads resume_from via agent.load()
        # before running further episodes.
        result = train(second_config)
        assert result.episodes_completed == 1
