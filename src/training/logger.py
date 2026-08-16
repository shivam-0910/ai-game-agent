# Source of truth:
# docs/phase_1_2_technical_specification.md
#
# Based on:
# src/training/train.py
#
# Phase: 4 — DQN Training Pipeline

"""Episode-level training statistics and CSV logging.

Writes one row per episode to a CSV file (Section 10 of the Phase 4
instructions), so Phase 5/6 can later load and analyze the data without
re-running training. Nothing here performs evaluation or generates
graphs — it only records.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional


@dataclass
class EpisodeStats:
    """Statistics recorded for a single training episode.

    ``avg_loss`` is ``None`` (written as an empty CSV field, not a fake
    0.0) for any episode in which no learning update occurred yet —
    typically the earliest episodes, before the replay buffer reaches
    the agent's minimum-experience threshold (Section 13 of the Phase 4
    instructions).
    """

    episode: int
    reward: float
    score: int
    snake_length: int
    steps: int
    epsilon: float
    avg_loss: Optional[float]
    num_learning_updates: int
    termination_reason: Optional[str]


class TrainingLogger:
    """Appends :class:`EpisodeStats` rows to a CSV file as they occur.

    Used as a context manager so the underlying file handle is always
    closed, even if training raises partway through:

    >>> with TrainingLogger(path) as logger:
    ...     logger.log(stats)
    """

    _FIELDNAMES = [f.name for f in fields(EpisodeStats)]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._FIELDNAMES)
        self._writer.writeheader()

    def log(self, stats: EpisodeStats) -> None:
        row = {name: getattr(stats, name) for name in self._FIELDNAMES}
        # Represent "no value" as an empty CSV field rather than the
        # string "None", so downstream CSV/pandas parsing treats it as
        # a genuine missing value (NaN) instead of ambiguous text.
        row = {k: ("" if v is None else v) for k, v in row.items()}
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
