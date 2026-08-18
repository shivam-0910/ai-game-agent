# Source of truth:
# docs/phase_6_5_network_architecture_experiment_plan.md
#
# Based on:
# src/experiments/experiment_config.py
# configs/experiments/A0_network_control.json
# configs/experiments/A1_network_hidden_128.json
# configs/experiments/A2_network_hidden_512.json
#
# Phase: 6.5 — Network Architecture Optimization

"""Config-level tests for the Phase 6.5 network-architecture screening set.

These tests only exercise ``ExperimentConfig.from_json`` and basic
attribute checks (Section: "each new experiment config loads
successfully" / "each experiment changes only network_hidden_size
relative to the control"). No training or evaluation runs here -- see
``docs/phase_6_5_network_architecture_experiment_plan.md`` for why actual
100/500-episode runs are deliberately out of scope for this phase's
preparation work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.experiment_config import ExperimentConfig

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs" / "experiments"

A_CONFIG_FILES = [
    "A0_network_control.json",
    "A1_network_hidden_128.json",
    "A2_network_hidden_512.json",
]

EXPECTED_HIDDEN_SIZES = {
    "A0_network_control": 256,
    "A1_network_hidden_128": 128,
    "A2_network_hidden_512": 512,
}


@pytest.fixture(params=A_CONFIG_FILES)
def a_config(request) -> ExperimentConfig:
    return ExperimentConfig.from_json(CONFIGS_DIR / request.param)


class TestArchitectureConfigsLoad:
    def test_config_loads_without_error(self, a_config: ExperimentConfig):
        assert isinstance(a_config, ExperimentConfig)

    def test_category_is_architecture(self, a_config: ExperimentConfig):
        assert a_config.category == "architecture"

    def test_parameter_is_network_hidden_size(self, a_config: ExperimentConfig):
        assert a_config.parameter == "network_hidden_size"

    def test_network_hidden_size_present(self, a_config: ExperimentConfig):
        assert a_config.network_hidden_size is not None

    def test_network_hidden_size_matches_expected_value(
        self, a_config: ExperimentConfig
    ):
        expected = EXPECTED_HIDDEN_SIZES[a_config.experiment_id]
        assert a_config.network_hidden_size == expected

    def test_experimental_value_field_matches_network_hidden_size(self, a_config: ExperimentConfig):
        assert int(a_config.experimental_value) == a_config.network_hidden_size


class TestArchitectureConfigsHoldConstantsFixed:
    """Section: "held constant" -- REWARD_DEATH and all Phase 6.1-6.4
    adopted hyperparameters must match the adopted configuration in every
    A-series experiment, screening and control alike."""

    def test_reward_death_is_adopted_value(self, a_config: ExperimentConfig):
        assert a_config.reward_overrides is not None
        assert a_config.reward_overrides["REWARD_DEATH"] == pytest.approx(-5.0)

    def test_epsilon_decay_is_adopted_value(self, a_config: ExperimentConfig):
        assert a_config.agent_overrides is not None
        assert a_config.agent_overrides["epsilon_decay"] == pytest.approx(0.995)

    def test_learning_rate_is_adopted_value(self, a_config: ExperimentConfig):
        assert a_config.agent_overrides is not None
        assert a_config.agent_overrides["learning_rate"] == pytest.approx(0.001)

    def test_gamma_is_adopted_value(self, a_config: ExperimentConfig):
        assert a_config.agent_overrides is not None
        assert a_config.agent_overrides["gamma"] == pytest.approx(0.95)

    def test_replay_capacity_is_adopted_value(self, a_config: ExperimentConfig):
        assert a_config.agent_overrides is not None
        assert a_config.agent_overrides["replay_capacity"] == 100000

    def test_batch_size_is_adopted_value(self, a_config: ExperimentConfig):
        assert a_config.agent_overrides is not None
        assert a_config.agent_overrides["batch_size"] == 64

    def test_is_not_marked_as_validation_run(self, a_config: ExperimentConfig):
        """These are screening configs; no A-series VALIDATION config
        exists yet, matching Phase 6.1-6.4's staged pattern."""
        assert a_config.is_validation_run is False


class TestArchitectureControlMatchesBaseline:
    def test_a0_baseline_and_experimental_value_are_equal(self):
        """The control (A0) must set experimental_value == baseline_value
        == the current hidden size (256), exactly mirroring how
        previous phase controls were defined."""
        cfg = ExperimentConfig.from_json(CONFIGS_DIR / "A0_network_control.json")
        assert cfg.baseline_value == cfg.experimental_value == "256"

    def test_a0_network_hidden_size_is_256(self):
        cfg = ExperimentConfig.from_json(CONFIGS_DIR / "A0_network_control.json")
        assert cfg.network_hidden_size == 256


class TestScreeningSetVariesOnlyNetworkHiddenSize:
    def test_all_three_experiments_have_distinct_hidden_sizes(self):
        configs = [
            ExperimentConfig.from_json(CONFIGS_DIR / fname) for fname in A_CONFIG_FILES
        ]
        hidden_sizes = [c.network_hidden_size for c in configs]
        assert len(set(hidden_sizes)) == 3

    def test_all_three_experiments_share_every_other_setting(self):
        """Beyond network_hidden_size, every other field that could affect
        training must be identical across A0-A2 so network_hidden_size is the
        only isolated variable (Phase 6.5 objective)."""
        configs = [
            ExperimentConfig.from_json(CONFIGS_DIR / fname) for fname in A_CONFIG_FILES
        ]
        for cfg in configs[1:]:
            assert cfg.seed == configs[0].seed
            assert cfg.training_episodes == configs[0].training_episodes
            assert cfg.evaluation_episodes == configs[0].evaluation_episodes
            assert cfg.evaluation_base_seed == configs[0].evaluation_base_seed
            assert cfg.reward_overrides == configs[0].reward_overrides
            assert cfg.agent_overrides == configs[0].agent_overrides

    def test_each_candidate_changes_only_hidden_size_relative_to_a0(self):
        """Verify that A1 and A2 each change only network_hidden_size
        relative to A0, and no other parameters differ."""
        a0 = ExperimentConfig.from_json(CONFIGS_DIR / "A0_network_control.json")
        a0_hidden = a0.network_hidden_size

        a1 = ExperimentConfig.from_json(CONFIGS_DIR / "A1_network_hidden_128.json")
        assert a1.network_hidden_size != a0_hidden
        assert a1.agent_overrides == a0.agent_overrides
        assert a1.reward_overrides == a0.reward_overrides

        a2 = ExperimentConfig.from_json(CONFIGS_DIR / "A2_network_hidden_512.json")
        assert a2.network_hidden_size != a0_hidden
        assert a2.agent_overrides == a0.agent_overrides
        assert a2.reward_overrides == a0.reward_overrides

    def test_all_three_experiments_have_unique_ids(self):
        configs = [
            ExperimentConfig.from_json(CONFIGS_DIR / fname) for fname in A_CONFIG_FILES
        ]
        ids = [c.experiment_id for c in configs]
        assert len(set(ids)) == 3

    def test_all_three_experiments_have_unique_model_paths(self):
        configs = [
            ExperimentConfig.from_json(CONFIGS_DIR / fname) for fname in A_CONFIG_FILES
        ]
        paths = [c.model_path for c in configs]
        assert len(set(paths)) == 3

    def test_all_three_experiments_have_unique_results_dirs(self):
        configs = [
            ExperimentConfig.from_json(CONFIGS_DIR / fname) for fname in A_CONFIG_FILES
        ]
        dirs = [c.results_dir for c in configs]
        assert len(set(dirs)) == 3
