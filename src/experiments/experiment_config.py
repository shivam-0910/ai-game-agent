# Source of truth:
# docs/phase_1_2_technical_specification.md
#
# Phase: 6 — Optimization and Experiments

"""Lightweight representation of a single controlled Phase 6 experiment.

Deliberately minimal (Section 6.0 of the Phase 6 instructions: "do NOT
build an overengineered experiment-management platform"). One
``ExperimentConfig`` fully describes one experiment: which single
parameter is changed, its baseline and experimental values, and where
the resulting model/results should be written. A JSON file per
experiment is optional but supported via ``from_json``/``save_json``,
matching the pattern already used by ``TrainingConfig``
(``src/training/config.py``).

This module owns only experiment *bookkeeping*. It does not itself run
training or evaluation — see ``src/experiments/run_experiment.py``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ExperimentConfig:
    """Full description of a single, single-variable experiment.

    Parameters
    ----------
    experiment_id:
        Short unique identifier, e.g. ``"R1_reward_food"``.
    name:
        Human-readable short name.
    description:
        One or two sentences explaining what is being tested and why
        (Section 6.1 of the Phase 6 instructions requires each reward
        experiment to be justified, not chosen arbitrarily).
    category:
        One of ``"reward"``, ``"epsilon"``, ``"learning_rate"``,
        ``"secondary"``, ``"architecture"``. Matches the Phase 6
        staged optimization order (Section 6.0-6.5).
    parameter:
        Name of the single changed variable, e.g. ``"REWARD_FOOD"`` or
        ``"epsilon_decay"``.
    baseline_value:
        The parameter's value in the baseline configuration. Stored as
        a string for uniform CSV/JSON round-tripping regardless of the
        underlying type (float, int, ...).
    experimental_value:
        The parameter's value for this experiment.
    seed:
        Master training seed for this experiment run.
    training_episodes:
        Number of training episodes for this run. Screening
        experiments use a smaller budget than a full validation run;
        the caller is responsible for labeling which is which (see
        ``is_validation_run``).
    evaluation_episodes:
        Number of evaluation episodes, using the Phase 5 evaluator.
        Should match the baseline protocol (100) for a fair comparison
        unless there is a documented reason to deviate.
    evaluation_base_seed:
        Base seed passed to the Phase 5 evaluator. Should match the
        baseline's evaluation seed (1,000,000) so every experiment is
        assessed on the same evaluation episodes.
    model_path:
        Where this experiment's trained checkpoint is saved. Must be
        distinct from ``models/best_model.pth`` (the baseline) and
        from every other experiment's path.
    results_dir:
        Directory where this experiment's training log and evaluation
        artifacts are written.
    is_validation_run:
        False for a short screening run, True for a full validation
        run of a promising screening candidate. Recorded so the
        registry never conflates the two (Section: "do not declare an
        improvement based on a noisy short run alone").
    reward_overrides:
        Passed straight through to ``SnakeEnv(reward_overrides=...)``
        when ``category == "reward"``. ``None`` for non-reward
        experiments.
    agent_overrides:
        Keyword overrides passed to the ``DQNAgent`` constructor (e.g.
        ``{"epsilon_decay": 0.990}`` or ``{"learning_rate": 5e-4}``)
        for non-reward experiments. ``None`` for reward experiments,
        which instead use ``reward_overrides``.
    network_hidden_size:
        Overrides ``DQNNetwork``'s hidden layer width for architecture
        experiments (Section 6.5). ``None`` uses the default (256).
    """

    experiment_id: str
    name: str
    description: str
    category: str
    parameter: str
    baseline_value: str
    experimental_value: str
    seed: int = 42
    training_episodes: int = 100
    evaluation_episodes: int = 100
    evaluation_base_seed: int = 1_000_000
    is_validation_run: bool = False
    model_path: str = ""
    results_dir: str = ""
    reward_overrides: Optional[dict[str, float]] = None
    agent_overrides: Optional[dict[str, Any]] = None
    network_hidden_size: Optional[int] = None

    _VALID_CATEGORIES = {"reward", "epsilon", "learning_rate", "secondary", "architecture"}

    def __post_init__(self) -> None:
        if self.category not in self._VALID_CATEGORIES:
            raise ValueError(
                f"Unknown category {self.category!r}; must be one of "
                f"{sorted(self._VALID_CATEGORIES)}."
            )
        if not self.model_path:
            self.model_path = f"models/experiments/{self.experiment_id}/model.pth"
        if not self.results_dir:
            self.results_dir = f"results/experiments/{self.experiment_id}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid_fields = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - valid_fields
        if unknown:
            raise ValueError(f"Unknown experiment config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
