# AI Game Agent — Reinforcement Learning Snake

## Overview

A Deep Q-Network (DQN) agent that learns to play Snake through reinforcement learning. The agent uses a compact feature-based state representation and epsilon-greedy exploration to learn effective Snake-playing strategies.

## Features

- Custom Gymnasium-compatible Snake environment with configurable board size
- DQN agent with experience replay, target network, and Huber loss
- Hyperparameter optimization across reward, exploration, learning rate, secondary hyperparameters, and network architecture
- Comprehensive testing with 426 unit and integration tests
- Reproducible training with deterministic seeding

## How It Works

**Snake Environment:** A custom 20×20 grid environment where the snake moves, eats food, grows, and terminates on collision.

**State:** The agent observes an 11-feature vector:
- 3 danger flags (straight, left, right relative to current heading)
- 4 heading one-hot vectors (up, down, left, right)
- 4 food-direction flags (food position relative to head)

**DQN:** A neural network with architecture 11 → 256 → 256 → 3 outputs. It learns Q-values for each action.

**Action:** The action space is relative: [Straight, Turn Left, Turn Right]. This eliminates instant self-collision by reversing direction.

**Reward:** The agent receives +10 for eating food, -5 for death (collision), and small shaping rewards for moving toward/away from food.

**Experience Replay:** Transitions are stored in a 100,000-capacity buffer and sampled in batches of 64 for training.

**Learning:** The agent uses a target network (hard update every 1,000 steps), Adam optimizer with SmoothL1 loss, and epsilon-greedy exploration (decay 0.995).

## Technology Stack

- Python 3.11+
- PyTorch 2.0+
- NumPy 1.24+
- Gymnasium 0.29+
- Matplotlib 3.7+
- pytest 7.4+

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
├── docs/                # Technical specification
└── requirements.txt
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the Project

**Run tests:**
```powershell
pytest -q
```

**Evaluate the final trained model:**
```powershell
python -m src.evaluation.run_evaluation --checkpoint models/best_model.pth --episodes 100
```

**Train a new model:**
```powershell
python -m src.training.run_training --config configs/baseline.json
```

## Final Model

The canonical final model is located at:
- **models/best_model.pth**

This is the promoted R2_reward_death_VALIDATION checkpoint (SHA256: `218688571ACF2337D87403B030D15C756B1E4332A62A9FB77A362295E5A96894`).

## Final Configuration

| Parameter | Value |
|-----------|-------|
| REWARD_DEATH | -5.0 |
| epsilon_decay | 0.995 |
| learning_rate | 0.001 |
| gamma | 0.95 |
| replay_capacity | 100000 |
| batch_size | 64 |
| network_hidden_size | 256 |

## Optimization Summary

**Phase 6.1 (Reward):** REWARD_DEATH = -5.0 was adopted. Validation average score = 34.13 (baseline: 29.26).

**Phase 6.2 (Epsilon Decay):** Alternative decay rates (0.990, 0.997, 0.999) were evaluated and rejected. Baseline epsilon_decay = 0.995 retained.

**Phase 6.3 (Learning Rate):** Alternative learning rates (0.0005, 0.002, 0.005) were evaluated and rejected. Baseline learning_rate = 0.001 retained.

**Phase 6.4 (Secondary Hyperparameters):** Gamma, replay capacity, and batch size alternatives were evaluated. The validation candidate (gamma=0.90) scored 33.13, failing to exceed the Phase 6.1 reference. Baseline values retained.

**Phase 6.5 (Network Architecture):** Alternative hidden layer sizes (128, 512) were evaluated. The validation candidate (hidden=128) scored 32.30, failing to exceed the Phase 6.1 reference. Baseline 256 retained.

**Final Decision:** Only REWARD_DEATH changed from the original baseline. All other parameters remained at their baseline values.

## Results

**Baseline reference score:** 29.26

**Final optimized score:** 34.13

**Improvement:** +4.87 (+16.6%)

## Testing

The repository contains 426 automated tests covering environment, agent, training, experiments, configuration, and evaluation. All tests pass.

## Limitations

- No graphical rendering (environment is headless with text-only ANSI rendering for debugging)
- No human keyboard control
- No deployment pipeline
- Evaluation uses a fixed seed set for reproducibility, not multi-seed averaging

## License

MIT License
