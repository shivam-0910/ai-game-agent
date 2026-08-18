# Source of truth:
# docs/phase_6_5_network_architecture_experiment_plan.md
#
# Based on:
# src/experiments/experiment_config.py
# configs/experiments/A1_network_hidden_128_VALIDATION.json
#
# Phase: 6.5 — Network Architecture Optimization

"""Config-level test for the A1 validation configuration.

Verifies the validation config structure and that it correctly
represents a 500-episode validation run of the A1 screening candidate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.experiment_config import ExperimentConfig

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs" / "experiments"
VALIDATION_CONFIG = "A1_network_hidden_128_VALIDATION.json"


@pytest.fixture
def a1_validation() -> ExperimentConfig:
    return ExperimentConfig.from_json(CONFIGS_DIR / VALIDATION_CONFIG)


class TestA1ValidationConfig:
    def test_config_loads_without_error(self, a1_validation: ExperimentConfig):
        assert isinstance(a1_validation, ExperimentConfig)

    def test_experiment_id_is_correct(self, a1_validation: ExperimentConfig):
        assert a1_validation.experiment_id == "A1_network_hidden_128_VALIDATION"

    def test_category_is_architecture(self, a1_validation: ExperimentConfig):
        assert a1_validation.category == "architecture"

    def test_parameter_is_network_hidden_size(self, a1_validation: ExperimentConfig):
        assert a1_validation.parameter == "network_hidden_size"

    def test_network_hidden_size_is_128(self, a1_validation: ExperimentConfig):
        assert a1_validation.network_hidden_size == 128

    def test_baseline_value_is_256(self, a1_validation: ExperimentConfig):
        assert a1_validation.baseline_value == "256"

    def test_experimental_value_is_128(self, a1_validation: ExperimentConfig):
        assert a1_validation.experimental_value == "128"

    def test_training_episodes_is_500(self, a1_validation: ExperimentConfig):
        assert a1_validation.training_episodes == 500

    def test_evaluation_episodes_is_100(self, a1_validation: ExperimentConfig):
        assert a1_validation.evaluation_episodes == 100

    def test_seed_is_42(self, a1_validation: ExperimentConfig):
        assert a1_validation.seed == 42

    def test_evaluation_base_seed_is_1000000(self, a1_validation: ExperimentConfig):
        assert a1_validation.evaluation_base_seed == 1000000

    def test_is_marked_as_validation_run(self, a1_validation: ExperimentConfig):
        assert a1_validation.is_validation_run is True

    def test_model_path_is_correct(self, a1_validation: ExperimentConfig):
        assert a1_validation.model_path == "models/experiments/A1_network_hidden_128_VALIDATION/model.pth"

    def test_results_dir_is_correct(self, a1_validation: ExperimentConfig):
        assert a1_validation.results_dir == "results/experiments/A1_network_hidden_128_VALIDATION"

    def test_reward_death_is_adopted_value(self, a1_validation: ExperimentConfig):
        assert a1_validation.reward_overrides is not None
        assert a1_validation.reward_overrides["REWARD_DEATH"] == pytest.approx(-5.0)

    def test_epsilon_decay_is_adopted_value(self, a1_validation: ExperimentConfig):
        assert a1_validation.agent_overrides is not None
        assert a1_validation.agent_overrides["epsilon_decay"] == pytest.approx(0.995)

    def test_learning_rate_is_adopted_value(self, a1_validation: ExperimentConfig):
        assert a1_validation.agent_overrides is not None
        assert a1_validation.agent_overrides["learning_rate"] == pytest.approx(0.001)

    def test_gamma_is_adopted_value(self, a1_validation: ExperimentConfig):
        assert a1_validation.agent_overrides is not None
        assert a1_validation.agent_overrides["gamma"] == pytest.approx(0.95)

    def test_replay_capacity_is_adopted_value(self, a1_validation: ExperimentConfig):
        assert a1_validation.agent_overrides is not None
        assert a1_validation.agent_overrides["replay_capacity"] == 100000

    def test_batch_size_is_adopted_value(self, a1_validation: ExperimentConfig):
        assert a1_validation.agent_overrides is not None
        assert a1_validation.agent_overrides["batch_size"] == 64
