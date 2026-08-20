# AI Game Agent — Reinforcement Learning Snake

## Overview

A Deep Q-Network (DQN) agent that learns to play Snake through reinforcement learning. The agent uses a compact feature-based state representation and epsilon-greedy exploration to learn effective Snake-playing strategies.

## Key Features

- **Custom Gymnasium-compatible Snake environment** with configurable board size
- **DQN agent** with experience replay, target network, and Huber loss
- **Hyperparameter optimization** across reward, exploration, learning rate, secondary hyperparameters, and network architecture
- **Comprehensive testing** with 426 unit and integration tests
- **Reproducible training** with deterministic seeding

## How the DQN Agent Works

The agent uses a compact 11-feature state representation:
- 3 danger flags (straight, left, right relative to current heading)
- 4 heading one-hot vectors (up, down, left, right)
- 4 food-direction flags (food position relative to head)

The action space is relative: [Straight, Turn Left, Turn Right], which eliminates the possibility of instant self-collision by reversing direction.

Training uses:
- Experience replay buffer (100,000 transitions)
- Separate target network (hard update every 1,000 steps)
- Epsilon-greedy exploration with decay
- Adam optimizer with SmoothL1 loss

## Project Structure

```
ai-game-agent/
├── src/
│   ├── environment/   # Snake environment (SnakeEnv)
│   ├── agent/         # DQN agent, network, replay buffer
│   ├── training/       # Training loop and configuration
│   ├── evaluation/     # Evaluation pipeline
│   └── experiments/    # Phase 6 experiment framework
├── models/              # Trained model checkpoints
├── results/             # Training logs and evaluation results
├── configs/             # Configuration files
├── tests/               # Unit and integration tests
├── docs/                # Technical specification and experiment results
└── requirements.txt
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running Tests

```powershell
pytest -q
```

## Running Evaluation

Evaluate the final trained model:

```powershell
python -m src.evaluation.run_evaluation --checkpoint models/best_model.pth --episodes 100
```

## Final Model

The canonical final model is located at:
- **models/best_model.pth**

This is the promoted R2_reward_death_VALIDATION checkpoint (SHA256: `218688571ACF2337D87403B030D15C756B1E4332A62A9FB77A362295E5A96894`).

## Final Adopted Configuration

```
REWARD_DEATH = -5.0
epsilon_decay = 0.995
learning_rate = 0.001
gamma = 0.95
replay_capacity = 100000
batch_size = 64
network_hidden_size = 256
```

## Optimization Results

**Phase 6.1 (Reward):** Selected REWARD_DEATH = -5.0. Validation average score = 34.13 (baseline: 29.26).

**Phase 6.2 (Epsilon Decay):** Alternative decay rates rejected after validation.

**Phase 6.3 (Learning Rate):** Alternative learning rates rejected after validation.

**Phase 6.4 (Secondary Hyperparameters):** Gamma, replay buffer size, and batch size alternatives rejected after validation.

**Phase 6.5 (Network Architecture):** Alternative hidden layer sizes (128, 512) rejected after validation.

The final adopted configuration remains the baseline configuration except for REWARD_DEATH = -5.0.

## Technology Stack

- Python 3.11+
- PyTorch 2.0+
- NumPy 1.24+
- Gymnasium 0.29+
- Matplotlib 3.7+
- pytest 7.4+

## Limitations

- No graphical rendering (environment is headless with text-only ANSI rendering for debugging)
- No human keyboard control
- No deployment pipeline
- Evaluation uses a fixed seed set for reproducibility, not multi-seed averaging
