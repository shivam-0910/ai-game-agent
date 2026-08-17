# Source of truth:
# docs/phase_1_2_technical_specification.md
#
# Based on:
# src/experiments/run_experiment.py
#
# Phase: 6 — Optimization and Experiments (CLI)

"""Tests for the Phase 6.2 CLI added to ``src.experiments.run_experiment``.

Only exercises argument parsing, error handling, and config loading.
Never triggers a real training/evaluation run: ``run_experiment_and_record``
is mocked wherever an experiment would actually execute.
"""

from __future__ import annotations

import json

import pytest

from src.experiments import run_experiment as run_experiment_module
from src.experiments.experiment_config import ExperimentConfig
from src.experiments.run_experiment import ExperimentRunResult, main


def _tiny_config_dict(**overrides) -> dict:
    base = dict(
        experiment_id="CLI_TEST",
        name="cli smoke test",
        description="Used only to test CLI parsing, never actually run.",
        category="epsilon",
        parameter="epsilon_decay",
        baseline_value="0.995",
        experimental_value="0.990",
    )
    base.update(overrides)
    return base


def _write_config(tmp_path, **overrides):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_tiny_config_dict(**overrides)))
    return path


def _fake_result(experiment_id: str) -> ExperimentRunResult:
    from pathlib import Path

    return ExperimentRunResult(
        experiment_id=experiment_id,
        training_episodes_completed=2,
        model_path=Path("models/experiments/fake/model.pth"),
        training_log_path=Path("results/experiments/fake/training_log.csv"),
        evaluation_results_csv=Path("results/experiments/fake/evaluation_results.csv"),
        evaluation_summary_json=Path("results/experiments/fake/evaluation_summary.json"),
        average_score=3.5,
        max_score=7,
        min_score=0,
        score_std=1.2,
        average_reward=4.4,
        average_steps=55.0,
        average_snake_length=4.5,
        termination_counts={"wall_collision": 2},
        elapsed_seconds=0.01,
    )


class TestHelp:
    def test_help_exits_zero_and_shows_usage(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "--config" in captured.out
        assert "--status" in captured.out


class TestArgumentErrors:
    def test_missing_config_is_argparse_error(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "--config" in captured.err

    def test_nonexistent_config_path_produces_clear_error(self, tmp_path, capsys):
        missing_path = tmp_path / "does_not_exist.json"
        exit_code = main(["--config", str(missing_path)])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()

    def test_invalid_json_produces_clear_error(self, tmp_path, capsys):
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("{not valid json")
        exit_code = main(["--config", str(bad_path)])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "invalid json" in captured.err.lower()

    def test_invalid_experiment_config_produces_clear_error(self, tmp_path, capsys):
        bad_path = tmp_path / "bad_config.json"
        bad_path.write_text(json.dumps(_tiny_config_dict(category="not_a_real_category")))
        exit_code = main(["--config", str(bad_path)])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "invalid experiment configuration" in captured.err.lower()


class TestValidConfigParsedWithoutRunning:
    def test_valid_config_parses_without_starting_real_experiment(self, tmp_path, monkeypatch):
        config_path = _write_config(tmp_path)
        called = {}

        def fake_run_experiment_and_record(experiment, status="screening", registry_path=None):
            called["experiment"] = experiment
            called["status"] = status
            return _fake_result(experiment.experiment_id)

        monkeypatch.setattr(
            run_experiment_module, "run_experiment_and_record", fake_run_experiment_and_record
        )

        exit_code = main(["--config", str(config_path)])

        assert exit_code == 0
        assert called["experiment"].experiment_id == "CLI_TEST"
        assert isinstance(called["experiment"], ExperimentConfig)

    def test_cli_passes_loaded_config_to_existing_runner(self, tmp_path, monkeypatch):
        config_path = _write_config(tmp_path, experiment_id="CLI_TEST_2")
        received = {}

        def fake_run_experiment_and_record(experiment, status="screening", registry_path=None):
            received["config"] = experiment
            return _fake_result(experiment.experiment_id)

        monkeypatch.setattr(
            run_experiment_module, "run_experiment_and_record", fake_run_experiment_and_record
        )

        main(["--config", str(config_path), "--status", "validated"])

        assert received["config"].experiment_id == "CLI_TEST_2"

    def test_status_argument_defaults_to_screening(self, tmp_path, monkeypatch):
        config_path = _write_config(tmp_path)
        received = {}

        def fake_run_experiment_and_record(experiment, status="screening", registry_path=None):
            received["status"] = status
            return _fake_result(experiment.experiment_id)

        monkeypatch.setattr(
            run_experiment_module, "run_experiment_and_record", fake_run_experiment_and_record
        )

        main(["--config", str(config_path)])
        assert received["status"] == "screening"

    def test_summary_output_contains_key_metrics(self, tmp_path, monkeypatch, capsys):
        config_path = _write_config(tmp_path)

        def fake_run_experiment_and_record(experiment, status="screening", registry_path=None):
            return _fake_result(experiment.experiment_id)

        monkeypatch.setattr(
            run_experiment_module, "run_experiment_and_record", fake_run_experiment_and_record
        )

        main(["--config", str(config_path)])
        captured = capsys.readouterr()
        assert "CLI_TEST" in captured.out
        assert "Average score" in captured.out
        assert "Max score" in captured.out
        assert "Min score" in captured.out
        assert "std dev" in captured.out
        assert "Elapsed time" in captured.out
        assert "Model path" in captured.out

    def test_execution_failure_is_reported_not_swallowed(self, tmp_path, monkeypatch, capsys):
        config_path = _write_config(tmp_path)

        def failing_run_experiment_and_record(experiment, status="screening", registry_path=None):
            raise RuntimeError("simulated training failure")

        monkeypatch.setattr(
            run_experiment_module, "run_experiment_and_record", failing_run_experiment_and_record
        )

        exit_code = main(["--config", str(config_path)])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "simulated training failure" in captured.err