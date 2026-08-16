"""DQN agent: owns the policy/target networks, optimizer, replay buffer,
and epsilon-greedy exploration, and implements the learning step.

Follows the baseline configuration in Section 6-10 / 16 of
``docs/phase_1_2_technical_specification.md``. No training loop lives
here (Section 5 of the Phase 3 instructions) — this module only exposes
the primitives a future training script will call.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn, optim

from src.agent.dqn_network import DQNNetwork, INPUT_SIZE, OUTPUT_SIZE
from src.agent.replay_buffer import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CAPACITY,
    DEFAULT_MIN_EXPERIENCES,
    ReplayBuffer,
)

# Baseline DQN hyperparameters (Section 6 / 16 of the technical spec).
# Treated as experimental starting values, not proven-optimal ones.
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_GAMMA = 0.95
DEFAULT_EPSILON_START = 1.0
DEFAULT_EPSILON_MIN = 0.01
DEFAULT_EPSILON_DECAY = 0.995  # multiplicative per-episode decay
DEFAULT_TARGET_UPDATE_FREQUENCY = 1_000  # training steps, hard update


class DQNAgent:
    """A Deep Q-Network agent for the 11-feature / 3-action Snake state.

    Owns the policy network (used for action selection and learning),
    a separate target network (used only to compute stable Bellman
    targets), the optimizer, and a replay buffer. Exposes the primitives
    a training loop needs (:meth:`select_action`, :meth:`remember`,
    :meth:`can_learn`, :meth:`learn`, :meth:`update_target_network`,
    :meth:`save`, :meth:`load`) without implementing the loop itself.
    """

    def __init__(
        self,
        state_size: int = INPUT_SIZE,
        action_size: int = OUTPUT_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        gamma: float = DEFAULT_GAMMA,
        epsilon_start: float = DEFAULT_EPSILON_START,
        epsilon_min: float = DEFAULT_EPSILON_MIN,
        epsilon_decay: float = DEFAULT_EPSILON_DECAY,
        replay_capacity: int = DEFAULT_CAPACITY,
        batch_size: int = DEFAULT_BATCH_SIZE,
        min_experiences: int = DEFAULT_MIN_EXPERIENCES,
        target_update_frequency: int = DEFAULT_TARGET_UPDATE_FREQUENCY,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if not 0.0 <= epsilon_min <= epsilon_start <= 1.0:
            raise ValueError("Require 0 <= epsilon_min <= epsilon_start <= 1.")

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.min_experiences = min_experiences
        self.target_update_frequency = target_update_frequency
        self._learn_step_count = 0

        # Device selection: prefer CUDA when available, but the agent
        # must work correctly (just slower) on CPU-only machines, since
        # CUDA is never a requirement for this project.
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        self.policy_network = DQNNetwork(state_size, output_size=action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, output_size=action_size).to(self.device)
        # Target network starts as an exact copy of the policy network.
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()  # target network is never trained directly

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        # Huber loss (SmoothL1Loss): more stable than plain MSE for DQN
        # because it is quadratic near zero error but linear for large
        # errors, which limits the impact of occasional large TD-error
        # outliers on the gradient (Section 6 / 10 of the spec, which
        # names Huber as the preferred, more stable alternative to MSE).
        self.loss_fn: nn.Module = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(capacity=replay_capacity, seed=seed)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Choose an action via epsilon-greedy exploration.

        With probability ``self.epsilon`` (only when ``training`` is
        True), a uniformly random action is returned. Otherwise the
        action with the highest predicted Q-value is returned.

        Parameters
        ----------
        state:
            Shape ``(state_size,)`` observation from the environment.
        training:
            When False, always acts greedily (epsilon is ignored) —
            used for evaluation-mode rollouts.
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        return self._greedy_action(state)

    def _greedy_action(self, state: np.ndarray) -> int:
        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32), device=self.device
        ).unsqueeze(0)
        self.policy_network.eval()
        with torch.no_grad():
            q_values = self.policy_network(state_tensor)
        self.policy_network.train()
        return int(torch.argmax(q_values, dim=1).item())

    def decay_epsilon(self) -> None:
        """Apply one step of the configured epsilon decay schedule.

        Intended to be called once per episode by the (future) training
        loop, per the "gradual decay over episodes" baseline strategy
        (Section 6/9 of the spec). Never decays below ``epsilon_min``.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition in the replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def can_learn(self) -> bool:
        """Whether enough experience has accumulated to start learning.

        Requires both the configured minimum-experience threshold and
        enough transitions to fill at least one batch (Section 7 of the
        spec: training should not start on a tiny, highly correlated
        buffer).
        """
        return len(self.replay_buffer) >= max(self.min_experiences, self.batch_size)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self) -> float:
        """Perform one gradient update from a sampled replay batch.

        Returns
        -------
        float
            The scalar loss value for this update (for logging by a
            future training loop).

        Raises
        ------
        RuntimeError
            If called before :meth:`can_learn` is True.
        """
        if not self.can_learn():
            raise RuntimeError(
                "learn() called before enough experiences were collected; "
                "check can_learn() first."
            )

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.as_tensor(batch.states, device=self.device)
        actions = torch.as_tensor(batch.actions, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.rewards, device=self.device)
        next_states = torch.as_tensor(batch.next_states, device=self.device)
        dones = torch.as_tensor(batch.dones, device=self.device)

        # Current Q-values for the actions actually taken.
        q_values = self.policy_network(states).gather(1, actions).squeeze(1)

        # Bellman target, computed from the target network with no
        # gradient flowing through it (target_network is frozen w.r.t.
        # this update; only policy_network is optimized).
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1).values
            # For terminal transitions (done=1), the bootstrapped future
            # value is zeroed out: target = reward, per Section 8.
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._learn_step_count += 1
        if self._learn_step_count % self.target_update_frequency == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self) -> None:
        """Hard-copy policy network weights into the target network.

        A full weight copy (rather than a soft/Polyak update) matches
        the baseline "hard update" decision in Section 8 of the spec.
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save policy/target network weights, optimizer state, and
        relevant training state (epsilon, learn-step count) via
        PyTorch's recommended state-dict approach."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy_network.state_dict(),
                "target_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "learn_step_count": self._learn_step_count,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load a checkpoint previously written by :meth:`save`.

        Restores both networks, the optimizer, and training state
        (epsilon, learn-step count) so training or evaluation can
        resume from exactly where it left off.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_network.load_state_dict(checkpoint["policy_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self._learn_step_count = checkpoint["learn_step_count"]
