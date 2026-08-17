# Source of truth:
# docs/phase_1_2_technical_specification.md
#
# Based on:
# src/experiments/experiment_config.py
# configs/experiments/E1_epsilon_decay_0990_VALIDATION.json
#
# Phase: 6.2 — Epsilon Decay Validation (preparation only)

"""Configuration-correctness tests for the E1 500-episode validation
run. Verifies the JSON loads through the existing ExperimentConfig
schema and resolves to the exact values specified by the Phase 6.2
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
    / "E1_epsilon_decay_0990_VALIDATION.json"
)


def _load() -> ExperimentConfig:
    return ExperimentConfig.from_json(CONFIG_PATH)


class TestE1ValidationConfigLoads:
    def test_config_file_exists(self):
        assert CONFIG_PATH.exists()

    def test_loads_successfully_through_experiment_config(self):
        config = _load()
        assert isinstance(config, ExperimentConfig)


class TestE1ValidationConfigValues:
    def test_experiment_id(self):
        assert _load().experiment_id == "E1_epsilon_decay_0990_VALIDATION"

    def test_training_episodes_is_500(self):
        assert _load().training_episodes == 500

    def test_evaluation_episodes_is_100(self):
        assert _load().evaluation_episodes == 100

    def test_seed_is_42(self):
        assert _load().seed == 42

    def test_evaluation_base_seed_is_1_000_000(self):
        assert _load().evaluation_base_seed == 1_000_000

    def test_is_validation_run_true(self):
        assert _load().is_validation_run is True

    def test_reward_overrides_reward_death_only(self):
        config = _load()
        assert config.reward_overrides == {"REWARD_DEATH": -5.0}

    def test_agent_overrides_epsilon_decay_only(self):
        config = _load()
        assert config.agent_overrides == {"epsilon_decay": 0.990}

    def test_category_and_parameter(self):
        config = _load()
        assert config.category == "epsilon"
        assert config.parameter == "epsilon_decay"

    def test_baseline_and_experimental_values(self):
        config = _load()
        assert config.baseline_value == "0.995"
        assert config.experimental_value == "0.990"


class TestE1ValidationConfigDoesNotDriftFromR2Protocol:
    """Confirms this config matches the Phase 6.1 R2 validation
    protocol on everything except epsilon_decay, per the requirement
    that E1 validation change only that one parameter."""

    def test_matches_r2_validation_seed_and_episode_protocol(self, tmp_path):
        r2_path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "experiments"
            / "R2_reward_death_VALIDATION.json"
        )
        if not r2_path.exists():
            import pytest

            pytest.skip("R2 validation config not present in this checkout")

        e1 = _load()
        r2 = ExperimentConfig.from_json(r2_path)

        assert e1.seed == r2.seed
        assert e1.training_episodes == r2.training_episodes
        assert e1.evaluation_episodes == r2.evaluation_episodes
        assert e1.evaluation_base_seed == r2.evaluation_base_seed
        assert e1.reward_overrides == r2.reward_overrides
        assert e1.is_validation_run == r2.is_validation_run


class TestE1ValidationConfigOutputPaths:
    def test_model_path_is_unique_to_this_experiment(self):
        config = _load()
        assert "E1_epsilon_decay_0990_VALIDATION" in config.model_path
        assert config.model_path != "models/best_model.pth"

    def test_results_dir_is_unique_to_this_experiment(self):
        config = _load()
        assert "E1_epsilon_decay_0990_VALIDATION" in config.results_dir
        assert config.results_dir != "results/evaluation"

    def test_does_not_collide_with_r2_validation_paths(self):
        config = _load()
        assert "R2_reward_death_VALIDATION" not in config.model_path
        assert "R2_reward_death_VALIDATION" not in config.results_dir
