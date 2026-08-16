"""Fixed-capacity experience replay buffer for the DQN agent.

Kept fully independent of any network/training logic (Section 4 of the
Phase 3 instructions) so it can be unit-tested in isolation and reused
unchanged if the network or training loop later changes.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Baseline replay hyperparameters (Section 7 of the technical spec).
DEFAULT_CAPACITY = 100_000
DEFAULT_BATCH_SIZE = 64
DEFAULT_MIN_EXPERIENCES = 1_000


@dataclass(frozen=True)
class Experience:
    """A single environment transition.

    ``done`` distinguishes terminal transitions (either ``terminated`` or
    ``truncated`` from the environment) from ongoing ones, so the agent's
    learning step can correctly zero out the bootstrapped next-state
    value for terminal transitions (Section 8 of the Phase 3 spec).
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass(frozen=True)
class Batch:
    """A sampled mini-batch, already stacked into arrays for easy
    tensor conversion by the agent."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """A fixed-capacity FIFO buffer of :class:`Experience` transitions.

    Once ``capacity`` is reached, adding a new experience discards the
    oldest one (a ``collections.deque`` with ``maxlen`` handles this
    automatically in O(1)).
    """

    def __init__(
        self, capacity: int = DEFAULT_CAPACITY, seed: int | None = None
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self._buffer: deque[Experience] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition, evicting the oldest if at capacity."""
        self._buffer.append(
            Experience(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
            )
        )

    def can_sample(self, batch_size: int) -> bool:
        """Whether at least ``batch_size`` experiences are available."""
        return len(self._buffer) >= batch_size

    def sample(self, batch_size: int = DEFAULT_BATCH_SIZE) -> Batch:
        """Uniformly sample a random mini-batch without replacement.

        Raises
        ------
        ValueError
            If fewer than ``batch_size`` experiences are stored. Callers
            are expected to check :meth:`can_sample` first; this is a
            safety net rather than the primary control-flow mechanism.
        """
        if not self.can_sample(batch_size):
            raise ValueError(
                f"Cannot sample {batch_size} experiences; only "
                f"{len(self._buffer)} available."
            )
        experiences: Sequence[Experience] = self._rng.sample(
            self._buffer, batch_size
        )
        return Batch(
            states=np.stack([e.state for e in experiences]),
            actions=np.array([e.action for e in experiences], dtype=np.int64),
            rewards=np.array([e.reward for e in experiences], dtype=np.float32),
            next_states=np.stack([e.next_state for e in experiences]),
            dones=np.array([e.done for e in experiences], dtype=np.float32),
        )
