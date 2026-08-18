# Source of truth:
# docs/phase_6_4_secondary_hyperparameters_experiment_plan.md
#
# Based on:
# src/experiments/experiment_config.py
# configs/experiments/S0_secondary_control.json
# configs/experiments/S1_gamma_090.json
# configs/experiments/S2_gamma_099.json
# configs/experiments/S3_replay_50000.json
# configs/experiments/S4_replay_200000.json
# configs/experiments/S5_batch_32.json
# configs/experiments/S6_batch_128.json
#
# Phase: 6.4 -- Secondary Hyperparameters Optimization

"""Config-level tests for the Phase 6.4 secondary-hyperparameters screening set.

These tests only exercise ``ExperimentConfig.from_json`` and basic
attribute checks (Section: "each new experiment config loads
successfully" / "each experiment changes only one secondary parameter
relative to the control"). No training or evaluation runs here -- see
``docs/phase_6_4_secondary_hyperparameters_experiment_plan.md`` for why actual
100/500-episode runs are deliberately out of scope for this phase's
preparation work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.experiment_config import ExperimentConfig

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs" / "experiments"

S_CONFIG_FILES = [
    "S0_secondary_control.json",
    "S1_gamma_090.json",
    "S2_gamma_099.json",
    "S3_replay_50000.json",
    "S4_replay_200000.json",
    "S5_batch_32.json",
    "S6_batch_128.json",
]

EXPECTED_SECONDARY_VALUES = {
    "S0_secondary_control": {"gamma": 0.95, "replay_capacity": 100000, "batch_size": 64},
    "S1_gamma_090": {"gamma": 0.90, "replay_capacity": 100000, "batch_size": 64},
    "S2_gamma_099": {"gamma": 0.99, "replay_capacity": 100000, "batch_size": 64},
    "S3_replay_50000": {"gamma": 0.95, "replay_capacity": 50000, "batch_size": 64},
    "S4_replay_200000": {"gamma": 0.95, "replay_capacity": 200000, "batch_size": 64},
    "S5_batch_32": {"gamma": 0.95, "replay_capacity": 100000, "batch_size": 32},
    "S6_batch_128": {"gamma": 0.95, "replay_capacity": 100000, "batch_size": 128},
}


@pytest.fixture(params=S_CONFIG_FILES)
def s_config(request) -> ExperimentConfig:
    return ExperimentConfig.from_json(CONFIGS_DIR / request.param)


class TestSecondaryConfigsLoad:
    def test_config_loads_without_error(self, s_config: ExperimentConfig):
        assert isinstance(s_config, ExperimentConfig)

    def test_category_is_secondary(self, s_config: ExperimentConfig):
        assert s_config.category == "secondary"

    def test_agent_overrides_present(self, s_config: ExperimentConfig):
        assert s_config.agent_overrides is not None
        assert "gamma" in s_config.agent_overrides
        assert "replay_capacity" in s_config.agent_overrides
        assert "batch_size" in s_config.agent_overrides

    def test_agent_overrides_secondary_values_match_expected(
        self, s_config: ExperimentConfig
    ):
        expected = EXPECTED_SECONDARY_VALUES[s_config.experiment_id]
        assert s_config.agent_overrides["gamma"] == pytest.approx(expected["gamma"])
        assert s_config.agent_overrides["replay_capacity"] == expected["replay_capacity"]
        assert s_config.agent_overrides["batch_size"] == expected["batch_size"]


class TestSecondaryConfigsHoldConstantsFixed:
    """Section: "held constant" -- REWARD_DEATH, epsilon_decay, and learning_rate
    must match the adopted Phase 6.1/6.2/6.3 configuration in every S-series
    experiment, screening and control alike."""

    def test_reward_death_is_adopted_value(self, s_config: ExperimentConfig):
        assert s_config.reward_overrides is not None
        assert s_config.reward_overrides["REWARD_DEATH"] == pytest.approx(-5.0)

    def test_epsilon_decay_is_adopted_value(self, s_config: ExperimentConfig):
        assert s_config.agent_overrides["epsilon_decay"] == pytest.approx(0.995)

    def test_learning_rate_is_adopted_value(self, s_config: ExperimentConfig):
        assert s_config.agent_overrides["learning_rate"] == pytest.approx(0.001)

    def test_network_hidden_size_is_unset(self, s_config: ExperimentConfig):
        assert s_config.network_hidden_size is None

    def test_is_not_marked_as_validation_run(self, s_config: ExperimentConfig):
        """These are screening configs; no S-series VALIDATION config
        exists yet, matching Phase 6.1/6.2/6.3's staged pattern."""
        assert s_config.is_validation_run is False


class TestSecondaryControlMatchesBaseline:
    def test_s0_has_control_parameter(self):
        """The control (S0) should have parameter set to indicate it's a control."""
        cfg = ExperimentConfig.from_json(CONFIGS_DIR / "S0_secondary_control.json")
        assert "control" in cfg.parameter.lower()

    def test_s0_secondary_values_are_baseline(self):
        """The control (S0) must have the baseline secondary values."""
        cfg = ExperimentConfig.from_json(CONFIGS_DIR / "S0_secondary_control.json")
        assert cfg.agent_overrides["gamma"] == pytest.approx(0.95)
        assert cfg.agent_overrides["replay_capacity"] == 100000
        assert cfg.agent_overrides["batch_size"] == 64


class TestScreeningSetVariesExactlyOneParameter:
    def test_all_seven_experiments_have_unique_ids(self):
        configs = [
            ExperimentConfig.from_json(CONFIGS_DIR / fname) for fname in S_CONFIG_FILES
        ]
        ids = [c.experiment_id for c in configs]
        assert len(set(ids)) == 7

    def test_all_seven_experiments_have_unique_model_paths(self):
        configs = [
            ExperimentConfig.from_json(CONFIGS_DIR / fname) for fname in S_CONFIG_FILES
        ]
        paths = [c.model_path for c in configs]
        assert len(set(paths)) == 7

    def test_all_seven_experiments_have_unique_results_dirs(self):
        configs = [
            ExperimentConfig.from_json(CONFIGS_DIR / fname) for fname in S_CONFIG_FILES
        ]
        dirs = [c.results_dir for c in configs]
        assert len(set(dirs)) == 7

    def test_all_experiments_share_training_evaluation_settings(self):
        """All S-series experiments must share the same training/evaluation
        protocol for fair comparison."""
        configs = [
            ExperimentConfig.from_json(CONFIGS_DIR / fname) for fname in S_CONFIG_FILES
        ]
        for cfg in configs:
            assert cfg.seed == 42
            assert cfg.training_episodes == 100
            assert cfg.evaluation_episodes == 100
            assert cfg.evaluation_base_seed == 1_000_000

    def test_each_candidate_changes_exactly_one_parameter_relative_to_s0(self):
        """Verify that S1-S6 each change exactly one secondary parameter
        relative to S0, and no other secondary parameters differ."""
        s0 = ExperimentConfig.from_json(CONFIGS_DIR / "S0_secondary_control.json")
        s0_gamma = s0.agent_overrides["gamma"]
        s0_replay = s0.agent_overrides["replay_capacity"]
        s0_batch = s0.agent_overrides["batch_size"]

        s1 = ExperimentConfig.from_json(CONFIGS_DIR / "S1_gamma_090.json")
        assert s1.agent_overrides["gamma"] != s0_gamma
        assert s1.agent_overrides["replay_capacity"] == s0_replay
        assert s1.agent_overrides["batch_size"] == s0_batch

        s2 = ExperimentConfig.from_json(CONFIGS_DIR / "S2_gamma_099.json")
        assert s2.agent_overrides["gamma"] != s0_gamma
        assert s2.agent_overrides["replay_capacity"] == s0_replay
        assert s2.agent_overrides["batch_size"] == s0_batch

        s3 = ExperimentConfig.from_json(CONFIGS_DIR / "S3_replay_50000.json")
        assert s3.agent_overrides["gamma"] == s0_gamma
        assert s3.agent_overrides["replay_capacity"] != s0_replay
        assert s3.agent_overrides["batch_size"] == s0_batch

        s4 = ExperimentConfig.from_json(CONFIGS_DIR / "S4_replay_200000.json")
        assert s4.agent_overrides["gamma"] == s0_gamma
        assert s4.agent_overrides["replay_capacity"] != s0_replay
        assert s4.agent_overrides["batch_size"] == s0_batch

        s5 = ExperimentConfig.from_json(CONFIGS_DIR / "S5_batch_32.json")
        assert s5.agent_overrides["gamma"] == s0_gamma
        assert s5.agent_overrides["replay_capacity"] == s0_replay
        assert s5.agent_overrides["batch_size"] != s0_batch

        s6 = ExperimentConfig.from_json(CONFIGS_DIR / "S6_batch_128.json")
        assert s6.agent_overrides["gamma"] == s0_gamma
        assert s6.agent_overrides["replay_capacity"] == s0_replay
        assert s6.agent_overrides["batch_size"] != s0_batch
