# Source of truth:
# docs/phase_6_3_learning_rate_experiment_plan.md
#
# Based on:
# src/experiments/experiment_config.py
# configs/experiments/L0_learning_rate_control.json
# configs/experiments/L1_learning_rate_0005.json
# configs/experiments/L2_learning_rate_002.json
# configs/experiments/L3_learning_rate_005.json
#
# Phase: 6.3 -- Learning Rate Optimization

"""Config-level tests for the Phase 6.3 learning-rate screening set.

These tests only exercise ``ExperimentConfig.from_json`` and basic
attribute checks (Section: "each new experiment config loads
successfully" / "each experiment changes only learning_rate relative
to the control"). No training or evaluation runs here -- see
``docs/phase_6_3_learning_rate_experiment_plan.md`` for why actual
100/500-episode runs are deliberately out of scope for this phase's
preparation work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.experiment_config import ExperimentConfig

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs" / "experiments"

L_CONFIG_FILES = [
    "L0_learning_rate_control.json",
    "L1_learning_rate_0005.json",
    "L2_learning_rate_002.json",
    "L3_learning_rate_005.json",
]

EXPECTED_LEARNING_RATES = {
    "L0_learning_rate_control": 0.001,
    "L1_learning_rate_0005": 0.0005,
    "L2_learning_rate_002": 0.002,
    "L3_learning_rate_005": 0.005,
}


@pytest.fixture(params=L_CONFIG_FILES)
def l_config(request) -> ExperimentConfig:
    return ExperimentConfig.from_json(CONFIGS_DIR / request.param)


class TestLearningRateConfigsLoad:
    def test_config_loads_without_error(self, l_config: ExperimentConfig):
        assert isinstance(l_config, ExperimentConfig)

    def test_category_is_learning_rate(self, l_config: ExperimentConfig):
        assert l_config.category == "learning_rate"

    def test_parameter_is_learning_rate(self, l_config: ExperimentConfig):
        assert l_config.parameter == "learning_rate"

    def test_agent_overrides_present(self, l_config: ExperimentConfig):
        assert l_config.agent_overrides is not None
        assert "learning_rate" in l_config.agent_overrides

    def test_agent_overrides_learning_rate_matches_expected_value(
        self, l_config: ExperimentConfig
    ):
        expected = EXPECTED_LEARNING_RATES[l_config.experiment_id]
        assert l_config.agent_overrides["learning_rate"] == pytest.approx(expected)

    def test_experimental_value_field_matches_agent_override(self, l_config: ExperimentConfig):
        assert float(l_config.experimental_value) == pytest.approx(
            l_config.agent_overrides["learning_rate"]
        )


class TestLearningRateConfigsHoldConstantsFixed:
    """Section: "held constant" -- REWARD_DEATH and epsilon_decay must
    match the adopted Phase 6.1/6.2 configuration in every L-series
    experiment, screening and control alike."""

    def test_reward_death_is_adopted_value(self, l_config: ExperimentConfig):
        assert l_config.reward_overrides is not None
        assert l_config.reward_overrides["REWARD_DEATH"] == pytest.approx(-5.0)

    def test_epsilon_decay_is_adopted_value(self, l_config: ExperimentConfig):
        assert l_config.agent_overrides["epsilon_decay"] == pytest.approx(0.995)

    def test_network_hidden_size_is_unset(self, l_config: ExperimentConfig):
        assert l_config.network_hidden_size is None

    def test_is_not_marked_as_validation_run(self, l_config: ExperimentConfig):
        """These are screening configs; no L-series VALIDATION config
        exists yet, matching Phase 6.1/6.2's staged pattern."""
        assert l_config.is_validation_run is False


class TestLearningRateControlMatchesBaseline:
    def test_l0_baseline_and_experimental_value_are_equal(self):
        """The control (L0) must set experimental_value == baseline_value
        == the currently adopted learning rate, exactly mirroring how
        E0 (Phase 6.2's epsilon control) was defined."""
        cfg = ExperimentConfig.from_json(CONFIGS_DIR / "L0_learning_rate_control.json")
        assert cfg.baseline_value == cfg.experimental_value == "0.001"


class TestScreeningSetOnlyVariesLearningRate:
    def test_all_four_experiments_have_distinct_learning_rates(self):
        configs = [
            ExperimentConfig.from_json(CONFIGS_DIR / fname) for fname in L_CONFIG_FILES
        ]
        lrs = [c.agent_overrides["learning_rate"] for c in configs]
        assert len(set(lrs)) == 4

    def test_all_four_experiments_share_every_other_setting(self):
        """Beyond learning_rate, every other field that could affect
        training must be identical across L0-L3 so learning_rate is the
        only isolated variable (Phase 6.3 objective)."""
        configs = [
            ExperimentConfig.from_json(CONFIGS_DIR / fname) for fname in L_CONFIG_FILES
        ]
        for cfg in configs[1:]:
            assert cfg.seed == configs[0].seed
            assert cfg.training_episodes == configs[0].training_episodes
            assert cfg.evaluation_episodes == configs[0].evaluation_episodes
            assert cfg.evaluation_base_seed == configs[0].evaluation_base_seed
            assert cfg.reward_overrides == configs[0].reward_overrides
            assert (
                cfg.agent_overrides["epsilon_decay"]
                == configs[0].agent_overrides["epsilon_decay"]
            )
            assert cfg.network_hidden_size == configs[0].network_hidden_size
