# Source of truth:
# docs/phase_1_2_technical_specification.md
#
# Based on:
# src/experiments/experiment_config.py
# configs/experiments/L2_learning_rate_002_VALIDATION.json
#
# Phase: 6.3 — Learning Rate Validation (preparation only)

"""Configuration-correctness tests for the L2 500-episode validation
run. Verifies the JSON loads through the existing ExperimentConfig
schema and resolves to the exact values specified by the Phase 6.3
validation protocol.

Deliberately does NOT call run_experiment() or
run_experiment_and_record() -- this file only checks that the
configuration is well-formed and correct. The actual 500-episode
training/evaluation is run later, manually, from PowerShell.
"""

from __future__ import annotations

from pathlib import Path

from src.experiments.experiment_config import ExperimentConfig

CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "experiments"
    / "L2_learning_rate_002_VALIDATION.json"
)


def _load() -> ExperimentConfig:
    return ExperimentConfig.from_json(CONFIG_PATH)


class TestL2ValidationConfigLoads:
    def test_config_file_exists(self):
        assert CONFIG_PATH.exists()

    def test_loads_successfully_through_experiment_config(self):
        config = _load()
        assert isinstance(config, ExperimentConfig)


class TestL2ValidationConfigValues:
    def test_experiment_id(self):
        assert _load().experiment_id == "L2_learning_rate_002_VALIDATION"

    def test_category_is_learning_rate(self):
        assert _load().category == "learning_rate"

    def test_parameter_is_learning_rate(self):
        assert _load().parameter == "learning_rate"

    def test_baseline_value_is_0_001(self):
        assert _load().baseline_value == "0.001"

    def test_experimental_value_is_0_002(self):
        assert _load().experimental_value == "0.002"

    def test_seed_is_42(self):
        assert _load().seed == 42

    def test_training_episodes_is_500(self):
        assert _load().training_episodes == 500

    def test_evaluation_episodes_is_100(self):
        assert _load().evaluation_episodes == 100

    def test_evaluation_base_seed_is_1_000_000(self):
        assert _load().evaluation_base_seed == 1_000_000

    def test_is_validation_run_true(self):
        assert _load().is_validation_run is True

    def test_agent_overrides_contains_learning_rate_0_002(self):
        config = _load()
        assert config.agent_overrides is not None
        assert config.agent_overrides.get("learning_rate") == 0.002

    def test_reward_overrides_remains_reward_death_minus_5(self):
        config = _load()
        assert config.reward_overrides == {"REWARD_DEATH": -5.0}

    def test_epsilon_decay_remains_0_995(self):
        config = _load()
        assert config.agent_overrides is not None
        assert config.agent_overrides.get("epsilon_decay") == 0.995


class TestL2ValidationConfigDoesNotDriftFromAdoptedProtocol:
    """Confirms this config matches the adopted Phase 6.1/6.2 protocol
    on everything except learning_rate, per the requirement that L2
    validation change only that one parameter."""

    def test_matches_e1_validation_seed_and_episode_protocol(self, tmp_path):
        e1_path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "experiments"
            / "E1_epsilon_decay_0990_VALIDATION.json"
        )
        if not e1_path.exists():
            import pytest

            pytest.skip("E1 validation config not present in this checkout")

        l2 = _load()
        e1 = ExperimentConfig.from_json(e1_path)

        assert l2.seed == e1.seed
        assert l2.training_episodes == e1.training_episodes
        assert l2.evaluation_episodes == e1.evaluation_episodes
        assert l2.evaluation_base_seed == e1.evaluation_base_seed
        assert l2.reward_overrides == e1.reward_overrides
        assert l2.is_validation_run == e1.is_validation_run


class TestL2ValidationConfigOutputPaths:
    def test_model_path_is_unique_to_this_experiment(self):
        config = _load()
        assert "L2_learning_rate_002_VALIDATION" in config.model_path
        assert config.model_path != "models/best_model.pth"

    def test_results_dir_is_unique_to_this_experiment(self):
        config = _load()
        assert "L2_learning_rate_002_VALIDATION" in config.results_dir
        assert config.results_dir != "results/evaluation"

    def test_does_not_collide_with_e1_validation_paths(self):
        config = _load()
        assert "E1_epsilon_decay_0990_VALIDATION" not in config.model_path
        assert "E1_epsilon_decay_0990_VALIDATION" not in config.results_dir

    def test_does_not_collide_with_r2_validation_paths(self):
        config = _load()
        assert "R2_reward_death_VALIDATION" not in config.model_path
        assert "R2_reward_death_VALIDATION" not in config.results_dir


class TestL2ValidationConfigDoesNotChangeUnrelatedHyperparameters:
    def test_network_hidden_size_is_null(self):
        config = _load()
        assert config.network_hidden_size is None

    def test_agent_overrides_contains_only_expected_keys(self):
        config = _load()
        assert config.agent_overrides is not None
        expected_keys = {"epsilon_decay", "learning_rate"}
        actual_keys = set(config.agent_overrides.keys())
        assert actual_keys == expected_keys

    def test_reward_overrides_contains_only_expected_keys(self):
        config = _load()
        assert config.reward_overrides is not None
        expected_keys = {"REWARD_DEATH"}
        actual_keys = set(config.reward_overrides.keys())
        assert actual_keys == expected_keys
