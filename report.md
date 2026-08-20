# AI Game Agent — Reinforcement Learning Snake

## Bright Hub Private Limited
## Artificial Intelligence Internship
## Project 4

## Table of Contents

1. Cover Page
2. Certificate
3. Acknowledgement
4. Abstract
5. Table of Contents
6. Introduction
7. Literature Review
8. Problem Statement
9. Objectives
10. Methodology
11. System Design
12. Implementation
13. Results
14. Conclusion
15. Future Scope
16. References

---

## Introduction

Reinforcement Learning (RL) has emerged as a powerful paradigm for training artificial agents to make sequential decisions in complex environments. Unlike supervised learning, which requires labeled examples, RL agents learn through trial and error by interacting with an environment and receiving rewards or penalties based on their actions. This approach is particularly well-suited for game-playing tasks, where the objective can be clearly defined through a reward function.

The game of Snake presents an ideal testbed for RL algorithms due to its balance of simplicity and strategic depth. The agent must learn to navigate a grid, collect food while growing, and avoid collisions with walls and its own body. The state space is sufficiently complex to require meaningful learning, yet tractable enough for a from-scratch implementation within an internship timeframe.

This project focuses on implementing a Deep Q-Network (DQN) agent for Snake. DQN extends traditional Q-learning by using a neural network to approximate the Q-function, enabling the agent to handle high-dimensional state spaces. The project includes the complete pipeline: environment design, state representation, action space definition, reward function design, DQN implementation, training, evaluation, and systematic hyperparameter optimization.

The work was completed as part of the Artificial Intelligence Internship at Bright Hub Private Limited, demonstrating practical application of modern RL techniques to a classic game environment.

---

## Literature Review

### Reinforcement Learning

Reinforcement Learning is a machine learning paradigm where an agent learns to make decisions by performing actions in an environment and receiving feedback in the form of rewards. The agent's objective is to maximize cumulative reward over time. Key concepts include the Markov Decision Process (MDP), which formalizes the interaction between agent and environment through states, actions, transition probabilities, and rewards.

### Q-Learning

Q-Learning is a model-free RL algorithm that learns the value of state-action pairs, known as Q-values. The Q-value represents the expected cumulative reward for taking a specific action in a given state and following an optimal policy thereafter. The algorithm updates Q-values using the Bellman equation, which incorporates the immediate reward and the discounted maximum Q-value of the next state.

### Deep Q-Networks (DQN)

Deep Q-Networks extend Q-learning by using a neural network to approximate the Q-function. This approach addresses the limitation of tabular Q-learning, which cannot handle large or continuous state spaces. DQN was introduced by Mnih et al. (2015) and demonstrated human-level performance on Atari games. Key innovations include experience replay, which breaks temporal correlations in training data, and a target network, which stabilizes learning by providing consistent Q-value targets.

### Exploration vs Exploitation

A fundamental challenge in RL is balancing exploration (trying new actions to discover better strategies) with exploitation (using known good actions to maximize reward). Epsilon-greedy is a common strategy where the agent selects a random action with probability epsilon and the best-known action otherwise. Epsilon is typically decayed over time to shift from exploration to exploitation.

### Experience Replay

Experience replay stores transitions (state, action, reward, next state, done) in a buffer and samples random mini-batches for training. This approach breaks temporal correlations, improves sample efficiency, and enables the reuse of past experiences. The replay buffer size is a critical hyperparameter that affects the diversity of training data.

### Target Networks

Target networks are copies of the main Q-network that are updated less frequently. They provide stable Q-value targets during training, reducing the risk of divergence caused by moving targets. Common update strategies include hard updates (periodic complete replacement) and soft updates (gradual interpolation).

This project implements DQN with experience replay and target networks, applying these techniques to the Snake environment.

---

## Problem Statement

The objective of this project is to develop an artificial intelligence agent capable of learning to play the game of Snake through reinforcement learning, without relying on hard-coded gameplay rules or human-designed strategies. The agent must learn optimal navigation, food collection, and collision avoidance strategies solely through interaction with the environment and reward feedback.

The challenge involves designing an appropriate state representation that captures relevant game information, defining an action space that enables effective control, crafting a reward function that guides the agent toward the objective, and implementing a DQN agent that can learn from experience. Additionally, the project requires systematic hyperparameter optimization to identify the best-performing configuration.

The final deliverable is a trained model that demonstrates improved performance over a baseline, along with a reproducible framework for training, evaluation, and experimentation.

---

## Objectives

The primary objectives of this project are:

1. **Build a custom Snake RL environment** that is Gymnasium-compatible, with configurable board size, deterministic seeding, and clear termination conditions.

2. **Define a compact state representation** using 11 features: danger flags, heading direction, and food direction relative to the snake head.

3. **Define a relative action space** of [Straight, Turn Left, Turn Right] to eliminate instant self-collision and simplify the learning problem.

4. **Design a reward function** that provides positive feedback for eating food, negative feedback for collisions, and optional shaping rewards to guide behavior.

5. **Implement a DQN agent** with experience replay, target network, epsilon-greedy exploration, and Huber loss for stable training.

6. **Train the agent** using the implemented environment and agent, with configurable training parameters and checkpointing.

7. **Evaluate performance** using a greedy policy with distinct seed sets to ensure reproducibility and fair comparison.

8. **Optimize important hyperparameters** through a systematic five-phase process covering reward, exploration, learning rate, secondary hyperparameters, and network architecture.

9. **Preserve a reproducible experiment framework** with configuration files, experiment registry, and detailed documentation.

10. **Produce a trained model** that represents the final optimized configuration and can be loaded for evaluation.

---

## Methodology

The project follows a structured pipeline from environment design to final optimization:

**Environment Design:** A custom Snake environment is implemented following Gymnasium conventions. The environment manages the game grid, snake movement, food spawning, collision detection, and reward calculation. It supports configurable board size, maximum episode steps, and reward overrides for experimentation.

**State Representation:** The state is an 11-dimensional vector containing 3 danger flags (straight, left, right relative to current heading), 4 heading one-hot encodings (up, down, left, right), and 4 food direction flags (food position relative to snake head). This compact representation enables fast network training.

**DQN Agent:** The agent uses a neural network with architecture 11 → 256 → 256 → 3. ReLU activations are used in hidden layers. The agent maintains a policy network and a target network, with the target network updated via hard replacement every 1,000 steps.

**Action Selection:** The agent uses epsilon-greedy exploration. During training, it selects a random action with probability epsilon (starting at 1.0 and decaying by 0.995 per episode) and the action with the highest Q-value otherwise. During evaluation, it uses a purely greedy policy (epsilon = 0).

**Experience Replay:** Transitions are stored in a replay buffer with capacity 100,000. During training, random mini-batches of size 64 are sampled for learning. This breaks temporal correlations and improves sample efficiency.

**Learning:** The agent uses the Adam optimizer with learning rate 0.001 and SmoothL1 (Huber) loss. The loss is computed between predicted Q-values and target Q-values calculated using the Bellman equation with discount factor gamma = 0.95.

**Training:** The training loop runs for a configurable number of episodes (typically 500). Each episode involves resetting the environment, running the agent until termination, decaying epsilon, and periodically saving checkpoints. The best model (highest score) is saved to `models/best_model.pth`.

**Evaluation:** Evaluation uses a greedy policy with a distinct seed set (starting from 1,000,000) to ensure reproducibility. Metrics include average score, max score, min score, standard deviation, and termination reasons.

**Optimization:** Hyperparameter optimization follows a five-phase process:
- Phase 6.1: Reward values (REWARD_DEATH tested at -5.0 vs baseline -10.0)
- Phase 6.2: Epsilon decay (0.990, 0.997, 0.999 vs baseline 0.995)
- Phase 6.3: Learning rate (0.0005, 0.002, 0.005 vs baseline 0.001)
- Phase 6.4: Secondary hyperparameters (gamma, replay capacity, batch size)
- Phase 6.5: Network architecture (hidden layer sizes 128, 512 vs baseline 256)

Each phase includes screening experiments (100 episodes) and validation experiments (500 episodes) for promising candidates. The experiment registry tracks all results.

---

## System Design

### Environment

The Snake environment (`SnakeEnv`) implements a 20×20 grid by default. The snake is represented as an ordered list of coordinates, with the head at index 0. Food spawns uniformly at random over empty cells. The environment supports:
- Configurable board size
- Optional maximum episode steps
- Reward overrides for experimentation
- Deterministic seeding via `reset(seed=...)`
- ANSI text rendering for debugging

The environment follows Gymnasium conventions with `reset()`, `step()`, and `close()` methods, returning observations, rewards, termination flags, truncation flags, and info dictionaries.

### State Representation

The state is an 11-feature vector:
- **Danger flags (3):** Boolean values indicating immediate collision risk straight ahead, to the left, and to the right relative to current heading.
- **Heading one-hot (4):** One-hot encoding of current direction (up, down, left, right).
- **Food direction (4):** Boolean flags indicating whether food is located up, down, left, or right relative to the snake head.

This compact representation enables fast training and is sufficient for the Snake task, as demonstrated by similar implementations in the literature.

### Action Space

The action space consists of three discrete actions:
- **Straight:** Continue in current direction
- **Turn Left:** Rotate 90 degrees counter-clockwise
- **Turn Right:** Rotate 90 degrees clockwise

Relative actions eliminate the possibility of instant self-collision by reversing direction, which is always fatal in Snake. This reduces the action space from 4 (absolute directions) to 3, simplifying the learning problem.

### Reward Function

The baseline reward function includes:
- **REWARD_FOOD:** +10 for eating food
- **REWARD_DEATH:** -10 for collision (wall or self)
- **REWARD_TOWARD_FOOD:** +0.1 for moving closer to food
- **REWARD_AWAY_FROM_FOOD:** -0.1 for moving away from food

Through optimization, **REWARD_DEATH** was changed from -10.0 to -5.0, which improved validation performance from 29.26 to 34.13. This change reduces the penalty for death, encouraging more exploration during training.

### DQN Architecture

The neural network architecture is:
- **Input layer:** 11 units (state features)
- **Hidden layer 1:** 256 units with ReLU activation
- **Hidden layer 2:** 256 units with ReLU activation
- **Output layer:** 3 units (Q-values for each action)

The network uses standard fully-connected layers. The output layer provides Q-values for each action without an activation function, allowing the network to learn both positive and negative Q-values.

### Experience Replay

The experience replay buffer has a capacity of 100,000 transitions. Each transition stores:
- State (11-dimensional vector)
- Action (integer 0-2)
- Reward (float)
- Next state (11-dimensional vector)
- Done flag (boolean)

During training, random mini-batches of size 64 are sampled. This breaks temporal correlations in the data and improves sample efficiency by reusing past experiences.

### Target Network

The target network is a copy of the policy network. It is updated via hard replacement every 1,000 steps. This provides stable Q-value targets during training, preventing the "moving target" problem that can cause divergence in Q-learning.

### Exploration

The agent uses epsilon-greedy exploration with the following parameters:
- **Initial epsilon:** 1.0 (fully random)
- **Epsilon decay:** 0.995 per episode
- **Minimum epsilon:** 0.01 (small exploration continues indefinitely)

During evaluation, epsilon is set to 0 for purely greedy action selection.

---

## Implementation

The implementation is organized into the following modules:

### `src/environment/`
- **`snake_env.py`:** Implements the `SnakeEnv` class with game logic, state extraction, reward calculation, and ANSI rendering.

### `src/agent/`
- **`dqn_agent.py`:** Implements the `DQNAgent` class with action selection, learning, experience replay, and model persistence.
- **`dqn_network.py`:** Implements the neural network architecture.
- **`replay_buffer.py`:** Implements the experience replay buffer.

### `src/training/`
- **`train.py`:** Implements the main training loop with episode management, checkpointing, and logging.
- **`config.py`:** Defines training configuration dataclass and JSON loading/saving.
- **`logger.py`:** Implements CSV logging of training metrics.
- **`run_training.py`:** Command-line entry point for training.

### `src/evaluation/`
- **`evaluate.py`:** Implements evaluation logic with greedy episodes and metric calculation.
- **`run_evaluation.py`:** Command-line entry point for evaluation.

### `src/experiments/`
- **`experiment_config.py`:** Defines experiment configuration dataclass for Phase 6 optimization.
- **`run_experiment.py`:** Implements experiment execution with training and evaluation.

### `tests/`
Comprehensive test suite covering:
- Environment mechanics and state extraction
- Agent action selection and learning
- Training loop functionality
- Experiment configuration validation
- Evaluation metrics calculation

The repository contains 426 tests, all passing.

### `configs/`
Configuration files for baseline training and Phase 6 experiments.

### `results/`
Training logs, evaluation results, and the experiment registry (`experiment_registry.csv`).

### `docs/`
Technical specification (`phase_1_2_technical_specification.md`) documenting the initial design decisions.

---

## Results

### Baseline Performance

The baseline configuration (before optimization) achieved:
- **Average score:** 29.26
- **Training episodes:** 500
- **Evaluation episodes:** 100
- **Configuration:** All parameters at initial baseline values (REWARD_DEATH = -10.0)

### Phase 6.1: Reward Optimization

**Experiment:** REWARD_DEATH tested at -5.0 vs baseline -10.0

**Screening (100 episodes):** Score 23.81 (baseline: 27.38)

**Validation (500 episodes):** Score 34.13 (baseline reference: 29.26)

**Decision:** Adopted REWARD_DEATH = -5.0

**Improvement:** +4.87 (+16.6%)

### Phase 6.2: Epsilon Decay Optimization

**Experiments tested:** 0.990, 0.997, 0.999 vs baseline 0.995

**Best candidate (0.990):** Validation score 32.13

**Decision:** Rejected (failed to exceed Phase 6.1 reference of 34.13)

**Result:** Baseline epsilon_decay = 0.995 retained

### Phase 6.3: Learning Rate Optimization

**Experiments tested:** 0.0005, 0.002, 0.005 vs baseline 0.001

**Best candidate (0.002):** Validation score 26.42

**Decision:** Rejected (failed to exceed Phase 6.1 reference)

**Result:** Baseline learning_rate = 0.001 retained

### Phase 6.4: Secondary Hyperparameters Optimization

**Experiments tested:** gamma (0.90, 0.99), replay_capacity (50,000, 200,000), batch_size (32, 128)

**Best candidate (gamma=0.90):** Validation score 33.13

**Decision:** Rejected (failed to exceed Phase 6.1 reference of 34.13)

**Result:** Baseline values retained (gamma=0.95, replay_capacity=100,000, batch_size=64)

### Phase 6.5: Network Architecture Optimization

**Experiments tested:** hidden layer sizes 128, 512 vs baseline 256

**Best candidate (hidden=128):** Validation score 32.30

**Decision:** Rejected (failed to exceed Phase 6.1 reference)

**Result:** Baseline network_hidden_size = 256 retained

### Optimization Summary

| Phase | Parameter Group | Result | Decision |
|-------|----------------|--------|----------|
| 6.1 | Reward | REWARD_DEATH=-5.0 improved validation to 34.13 | Adopted |
| 6.2 | Epsilon decay | Alternatives rejected (best: 32.13) | Baseline retained |
| 6.3 | Learning rate | Alternatives rejected (best: 26.42) | Baseline retained |
| 6.4 | Secondary hyperparameters | Validation failed to exceed reference (best: 33.13) | Baseline retained |
| 6.5 | Network architecture | Validation failed to exceed reference (best: 32.30) | Baseline retained |

### Final Model

The final model is located at `models/best_model.pth`. This is the promoted R2_reward_death_VALIDATION checkpoint, corresponding to the final adopted configuration with REWARD_DEATH = -5.0 and all other parameters at baseline values.

**SHA256:** `218688571ACF2337D87403B030D15C756B1E4332A62A9FB77A362295E5A96894`

### Testing

The repository contains 426 automated tests covering all major components. All tests pass, ensuring code correctness and regression prevention.

---

## Conclusion

This project successfully demonstrated the application of Deep Q-Networks to the classic game of Snake. A complete RL pipeline was implemented, including a custom Gymnasium-compatible environment, compact state representation, relative action space, reward function, DQN agent with experience replay and target network, training loop, evaluation protocol, and systematic hyperparameter optimization.

The optimization process identified that reducing the death penalty from -10.0 to -5.0 significantly improved performance, achieving a validation average score of 34.13 compared to the baseline of 29.26—a 16.6% improvement. All other hyperparameter alternatives failed to exceed this reference, resulting in a final adopted configuration that differs from the baseline only in the REWARD_DEATH parameter.

The project delivered a trained model, comprehensive test suite (426 tests), experiment registry tracking all optimization results, and detailed documentation. The work demonstrates practical application of modern RL techniques and provides a reproducible framework for future experimentation.

---

## Future Scope

Potential future improvements to this project include:

**Graphical Visualization:** Implement pygame or similar library to provide real-time visual rendering of the game, enabling human observation of agent behavior and potential interactive gameplay.

**Richer State Representation:** Explore grid-based or convolutional network approaches that capture full spatial information, potentially enabling better long-term planning as the snake grows longer.

**Additional RL Algorithms:** Implement and compare alternative algorithms such as Double DQN, Dueling DQN, or Proximal Policy Optimization (PPO) to benchmark performance.

**Extended Training:** Increase training episodes and explore curriculum learning or transfer learning approaches to further improve performance.

**Environment Complexity:** Add obstacles, multiple food items, or dynamic board sizes to increase challenge and test generalization.

**Deployment:** Package the trained model as a web service or standalone application for broader accessibility.

**Multi-Agent Scenarios:** Extend to multi-agent Snake where multiple snakes compete or cooperate in the same environment.

---

## References

1. Bright Hub Private Limited. Artificial Intelligence Internship Handbook.

2. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). "Human-level control through deep reinforcement learning." Nature, 518(7540), 529-533.

3. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.

4. Brockman, G., Cheung, V., Pettersson, L., et al. (2016). OpenAI Gym. arXiv preprint arXiv:1606.01540.

5. Towers, M., et al. (2023). Gymnasium: An API Standard for Reinforcement Learning Environments.

6. PyTorch Documentation. https://pytorch.org/docs/

7. Gymnasium Documentation. https://gymnasium.farama.org/

---

[Insert Screenshot: GitHub Repository]

[Insert Screenshot: Test Results]

[Insert Screenshot: Final Evaluation]

[Insert Screenshot: Project Structure]
