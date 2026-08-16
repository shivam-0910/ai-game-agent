"""Q-network used by the DQN agent.

Implements the baseline architecture frozen in Section 6 / Section 16 of
``docs/phase_1_2_technical_specification.md``:

    11 (state features) -> 256 -> 256 -> 3 (Q-values)

with ReLU activations between hidden layers and a linear (no activation)
output layer.
"""

from __future__ import annotations

import torch
from torch import nn

# Baseline architecture constants (Section 6 / 16 of the spec).
INPUT_SIZE = 11
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3


class DQNNetwork(nn.Module):
    """A small MLP mapping a Snake state vector to per-action Q-values.

    The output layer intentionally has no activation function (no
    softmax, no sigmoid): Q-values are estimates of expected discounted
    return, not probabilities, and are not constrained to sum to 1 or
    lie in [0, 1]. Squashing them would distort the Bellman targets the
    network is trained to regress against.
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        output_size: int = OUTPUT_SIZE,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a state (or batch of states).

        Parameters
        ----------
        state:
            Tensor of shape ``(input_size,)`` for a single state or
            ``(batch_size, input_size)`` for a batch.

        Returns
        -------
        torch.Tensor
            Raw Q-values of shape ``(output_size,)`` or
            ``(batch_size, output_size)`` matching the input rank.
        """
        return self.network(state)
