# Source of truth:
# docs/phase_1_2_technical_specification.md
#
# Based on:
# src/experiments/experiment_config.py
# src/experiments/run_experiment.py
#
# Phase: 6 — Optimization and Experiments

"""Test suite for the Phase 6 experiment-management code.

Uses tiny training/evaluation budgets (a handful of episodes each) so
the suite stays fast, per the Phase 6 instructions: "do not create
tests that perform hundreds of training episodes." No test depends on
the real ``models/best_model.pth``; every experiment here trains and
evaluates its own throwaway checkpoint under ``tmp_path``.
"""

from __future__ import annotations

import csv

import pytest

from src.environment.constants import REWARD_DEATH, REWARD_FOOD
from src.experiments.experiment_config import ExperimentConfig
from src.experiments.run_experiment import (
    append_to_registry,
    run_experiment,
    run_experiment_and_record,
)


def _tiny_reward_experiment(tmp_path, **overrides) -> ExperimentConfig:
    defaults = dict(
        experiment_id="TEST_R1",
        name="test reward food increase",
        description="Unit-test-scale reward experiment.",
        category="reward",
        parameter="REWARD_FOOD",
        baseline_value=str(REWARD_FOOD),
        experimental_value="15.0",
        seed=0,
        training_episodes=2,
        evaluation_episodes=2,
        evaluation_base_seed=999_000,
        model_path=str(tmp_path / "models" / "test_r1" / "model.pth"),
        results_dir=str(tmp_path / "results" / "test_r1"),
        reward_overrides={"REWARD_FOOD": 15.0},
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _tiny_agent_experiment(tmp_path, **overrides) -> ExperimentConfig:
    defaults = dict(
        experiment_id="TEST_E1",
        name="test epsilon decay",
        description="Unit-test-scale epsilon experiment.",
        category="epsilon",
        parameter="epsilon_decay",
        baseline_value="0.995",
        experimental_value="0.990",
        seed=0,
        training_episodes=2,
        evaluation_episodes=2,
        evaluation_base_seed=999_000,
        model_path=str(tmp_path / "models" / "test_e1" / "model.pth"),
        results_dir=str(tmp_path / "results" / "test_e1"),
        agent_overrides={"epsilon_decay": 0.990},
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


# ---------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------


class TestExperimentConfig:
    def test_loads_correctly_with_required_fields(self, tmp_path):
        config = _tiny_reward_experiment(tmp_path)
        assert config.experiment_id == "TEST_R1"
        assert config.category == "reward"
        assert config.reward_overrides == {"REWARD_FOOD": 15.0}

    def test_rejects_unknown_category(self, tmp_path):
        with pytest.raises(ValueError):
            _tiny_reward_experiment(tmp_path, category="not_a_real_category")

    def test_default_paths_are_derived_from_id_when_omitted(self):
        config = ExperimentConfig(
            experiment_id="R9",
            name="n",
            description="d",
            category="reward",
            parameter="REWARD_FOOD",
            baseline_value="10.0",
            experimental_value="12.0",
        )
        assert config.model_path == "models/experiments/R9/model.pth"
        assert config.results_dir == "results/experiments/R9"

    def test_json_round_trip(self, tmp_path):
        config = _tiny_reward_experiment(tmp_path)
        path = tmp_path / "config.json"
        config.save_json(path)
        loaded = ExperimentConfig.from_json(path)
        assert loaded.experiment_id == config.experiment_id
        assert loaded.reward_overrides == config.reward_overrides

    def test_json_rejects_unknown_keys(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(
            '{"experiment_id": "X", "name": "n", "description": "d", '
            '"category": "reward", "parameter": "p", "baseline_value": "1", '
            '"experimental_value": "2", "not_a_real_field": true}'
        )
        with pytest.raises(ValueError):
            ExperimentConfig.from_json(path)


# ---------------------------------------------------------------------
# Baseline preservation
# ---------------------------------------------------------------------


class TestBaselinePreservation:
    def test_baseline_reward_constants_unchanged_by_config_creation(self, tmp_path):
        _tiny_reward_experiment(tmp_path)
        # Module-level baseline constants must never be mutated just
        # by constructing experiment configs.
        assert REWARD_FOOD == 10.0
        assert REWARD_DEATH == -10.0

    def test_experiment_model_path_is_never_the_baseline_path(self, tmp_path):
        config = _tiny_reward_experiment(tmp_path)
        assert config.model_path != "models/best_model.pth"

    def test_experiment_results_dir_is_never_baseline_results_dir(self, tmp_path):
        config = _tiny_reward_experiment(tmp_path)
        assert config.results_dir != "results/evaluation"
        assert config.results_dir != "results/training"


# ---------------------------------------------------------------------
# Output path uniqueness
# ---------------------------------------------------------------------


class TestOutputPathUniqueness:
    def test_different_experiment_ids_get_different_default_paths(self):
        config_a = ExperimentConfig(
            experiment_id="R1",
            name="a",
            description="d",
            category="reward",
            parameter="REWARD_FOOD",
            baseline_value="10.0",
            experimental_value="15.0",
        )
        config_b = ExperimentConfig(
            experiment_id="R2",
            name="b",
            description="d",
            category="reward",
            parameter="REWARD_DEATH",
            baseline_value="-10.0",
            experimental_value="-5.0",
        )
        assert config_a.model_path != config_b.model_path
        assert config_a.results_dir != config_b.results_dir


# ---------------------------------------------------------------------
# Single-variable representation
# ---------------------------------------------------------------------


class TestSingleVariableRepresentation:
    def test_reward_experiment_only_sets_reward_overrides(self, tmp_path):
        config = _tiny_reward_experiment(tmp_path)
        assert config.reward_overrides is not None
        assert config.agent_overrides is None
        assert config.network_hidden_size is None

    def test_agent_experiment_only_sets_agent_overrides(self, tmp_path):
        config = _tiny_agent_experiment(tmp_path)
        assert config.agent_overrides == {"epsilon_decay": 0.990}
        assert config.reward_overrides is None
        assert config.network_hidden_size is None


# ---------------------------------------------------------------------
# Running experiments (tiny budgets)
# ---------------------------------------------------------------------


class TestRunExperiment:
    def test_reward_experiment_runs_end_to_end(self, tmp_path):
        config = _tiny_reward_experiment(tmp_path)
        result = run_experiment(config)
        assert result.model_path.exists()
        assert result.training_log_path.exists()
        assert result.evaluation_results_csv.exists()
        assert result.evaluation_summary_json.exists()
        assert result.training_episodes_completed == 2

    def test_agent_override_experiment_runs_end_to_end(self, tmp_path):
        config = _tiny_agent_experiment(tmp_path)
        result = run_experiment(config)
        assert result.model_path.exists()
        assert result.training_episodes_completed == 2

    def test_reward_override_actually_affects_training_env(self, tmp_path, monkeypatch):
        """Sanity check that the experiment's reward_overrides really
        reach SnakeEnv during training, not just during evaluation."""
        captured = {}
        from src.experiments import run_experiment as run_experiment_module

        original_build_env = run_experiment_module._build_env

        def spy_build_env(experiment, board_size=20):
            env = original_build_env(experiment, board_size=board_size)
            captured["reward_food"] = env.reward_food
            return env

        monkeypatch.setattr(run_experiment_module, "_build_env", spy_build_env)
        config = _tiny_reward_experiment(tmp_path)
        run_experiment(config)
        assert captured["reward_food"] == 15.0

    def test_network_hidden_size_override_changes_architecture(self, tmp_path):
        config = _tiny_reward_experiment(
            tmp_path,
            experiment_id="TEST_ARCH",
            category="architecture",
            parameter="hidden_size",
            baseline_value="256",
            experimental_value="128",
            reward_overrides=None,
            network_hidden_size=128,
            model_path=str(tmp_path / "models" / "test_arch" / "model.pth"),
            results_dir=str(tmp_path / "results" / "test_arch"),
        )
        result = run_experiment(config)
        import torch

        ckpt = torch.load(result.model_path, map_location="cpu", weights_only=True)
        # First linear layer weight shape is (hidden_size, input_size).
        assert ckpt["policy_state_dict"]["network.0.weight"].shape[0] == 128


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------


class TestRegistry:
    def test_registry_created_and_row_written(self, tmp_path):
        config = _tiny_reward_experiment(tmp_path)
        registry_path = tmp_path / "registry.csv"
        result = run_experiment(config)
        append_to_registry(config, result, status="screening", registry_path=registry_path)

        assert registry_path.exists()
        with open(registry_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["experiment_id"] == "TEST_R1"
        assert rows[0]["status"] == "screening"

    def test_registry_appends_without_overwriting(self, tmp_path):
        registry_path = tmp_path / "registry.csv"
        config_a = _tiny_reward_experiment(tmp_path, experiment_id="TEST_A")
        config_b = _tiny_reward_experiment(
            tmp_path,
            experiment_id="TEST_B",
            model_path=str(tmp_path / "models" / "test_b" / "model.pth"),
            results_dir=str(tmp_path / "results" / "test_b"),
        )
        run_experiment_and_record(config_a, status="screening", registry_path=registry_path)
        run_experiment_and_record(config_b, status="screening", registry_path=registry_path)

        with open(registry_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert {r["experiment_id"] for r in rows} == {"TEST_A", "TEST_B"}

    def test_registry_row_has_expected_columns(self, tmp_path):
        config = _tiny_reward_experiment(tmp_path)
        registry_path = tmp_path / "registry.csv"
        result = run_experiment(config)
        append_to_registry(config, result, status="screening", registry_path=registry_path)

        with open(registry_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames)
        assert {
            "experiment_id",
            "category",
            "parameter",
            "baseline_value",
            "experiment_value",
            "average_score",
            "max_score",
            "min_score",
            "score_std",
            "self_collision_rate",
            "wall_collision_rate",
            "status",
        }.issubset(fieldnames)

    def test_evaluation_results_associated_with_correct_experiment(self, tmp_path):
        """The evaluation results/summary paths returned for an
        experiment must live under that experiment's own results_dir,
        never a shared or baseline directory."""
        config = _tiny_reward_experiment(tmp_path)
        result = run_experiment(config)
        assert str(result.evaluation_results_csv).startswith(config.results_dir)
        assert str(result.evaluation_summary_json).startswith(config.results_dir)


# ---------------------------------------------------------------------
# Baseline files never touched
# ---------------------------------------------------------------------


class TestBaselineFilesUntouched:
    def test_does_not_write_to_real_models_dir(self, tmp_path):
        config = _tiny_reward_experiment(tmp_path)
        run_experiment(config)
        # Experiment model path must never coincide with the real
        # project's models/best_model.pth.
        assert "best_model.pth" not in str(config.model_path)

    def test_does_not_write_to_baseline_evaluation_dir(self, tmp_path):
        config = _tiny_reward_experiment(tmp_path)
        run_experiment(config)
        assert config.results_dir != "results/evaluation"
