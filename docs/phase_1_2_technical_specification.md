# Phase 1.2 — Technical Specification

**Project:** AI Game Agent — Reinforcement Learning Snake
**Phase:** 1.2 (Architecture & Technical Design Only)
**Status:** No code implemented. This document is a design blueprint for Phase 2.

---

## 1. Executive Summary

This document defines the technical design for a Deep Q-Network (DQN) agent
that learns to play Snake in a custom, Gymnasium-style environment. It
covers environment mechanics, state/action representation, reward design,
network architecture, training mechanics, evaluation, optimization
planning, testing strategy, and module responsibilities.

No implementation exists yet. All values given here (reward magnitudes,
network sizes, hyperparameters) are **initial baseline proposals**, subject
to change once real experiments are run in later phases. No performance
numbers in this document are results — they are targets/metrics
definitions only.

---

## 2. Environment Design

| Aspect | Decision | Justification |
|---|---|---|
| Board dimensions | Fixed grid, e.g. 20×20 cells | Large enough for non-trivial strategy, small enough to keep the state space and episode length manageable for a from-scratch DQN. Configurable via `configs/`, not hardcoded. |
| Grid representation | 2D integer grid internally (`0`=empty, `1`=snake body, `2`=snake head, `3`=food) | Simple, debuggable, and easy to render for visual inspection; also convertible into either state approach chosen in Section 3. |
| Snake representation | Ordered list/deque of `(row, col)` cells, head at index 0 | A deque gives O(1) append/pop for movement and growth, which is the natural operation set for a snake body. |
| Food representation | Single `(row, col)` cell, one food item at a time | Matches classic Snake rules; keeps the reward/state design simple. Multiple simultaneous food items are an optional future extension, not baseline. |
| Starting position | Snake spawns centered on the board, short initial length (e.g. 3 segments) in a straight line | Centered start avoids immediate wall proximity bias and gives the agent room to maneuver in any direction early on. |
| Starting direction | Fixed (e.g. facing right) unless randomized under a seed | A fixed default simplifies debugging; randomizing under a seed can be a later robustness improvement, not required for baseline. |
| Food spawning rules | Uniform random over all empty cells, re-rolled if it lands on the snake body | Guarantees food is always reachable in principle and avoids invalid spawns. |
| Collision rules | Wall collision = episode ends; self-collision = episode ends | Standard Snake rules; keeps the termination condition simple and unambiguous for reward shaping. |
| Episode termination | Collision (wall/self), OR step count exceeds a max-steps limit, OR board fully filled (win condition) | The step limit prevents infinite non-productive episodes (e.g., an agent that loops safely forever without eating), which is important during early training when the policy is near-random. |
| Score calculation | Score = number of food items eaten (i.e., snake length − starting length) | Directly reflects the task objective and is independent of reward-shaping choices, so it remains a stable evaluation metric even if the reward function changes later. |
| Step mechanics | Each step: read action → compute new head position → check collision → check food → update body → return `(state, reward, terminated, truncated, info)` | Matches the Gymnasium `step()` contract, keeping the environment compatible with standard RL tooling and conventions. |
| Immediate reversal | Disallowed at the environment/action level (see Section 4) | Reversing directly into your own neck is always a self-collision in classic Snake; disallowing it at the action level avoids wasting agent exploration on a guaranteed-fatal move and simplifies the action space. |
| Full-board handling | Treated as episode termination with a "win" flag in `info` | A full board is a natural success terminal state; there is no valid next move, so it must end the episode. |
| Randomness/seeding | Environment accepts a `seed` parameter (Gymnasium convention: `reset(seed=...)`) controlling food spawn RNG and optional randomized start | Required for reproducibility of both debugging and later experiment comparisons (Section 11). |

---

## 3. State Representation

Two candidate approaches are compared.

### Approach A — Compact feature-based state

A fixed-length numeric vector describing relevant relationships rather than
the raw grid, e.g.:
- Danger indicators (straight/left/right relative to current heading)
- Current movement direction (one-hot: up/down/left/right)
- Food location relative to head (e.g., food is left/right/up/down of head)

| Property | Assessment |
|---|---|
| State size | Small — typically 10-15 scalar features |
| Information available | Local danger + relative food direction; no full board layout |
| Computational cost | Very low; trivial to compute each step |
| Learning difficulty | Lower — small input space, faster convergence, less data needed |
| Advantages | Fast training, small network, easy to debug, works well for classic Snake-agent implementations |
| Disadvantages | Loses global information (e.g., body shape far from the head, long-term trap avoidance for a long snake); may plateau as the snake grows long |

### Approach B — Grid/board-based state

The full board (or a fixed-size window) as a 2D (or multi-channel) array
fed into a convolutional network.

| Property | Assessment |
|---|---|
| State size | Large — e.g., 20×20 (400+ cells), potentially multiple channels |
| Information available | Full spatial layout of snake body and food |
| Computational cost | Higher — requires CNN layers, more compute per step |
| Learning difficulty | Higher — larger input space, generally needs more training episodes and a larger network |
| Advantages | In principle captures long-term spatial structure and can generalize better as the snake grows |
| Disadvantages | Slower training, more complex network, more hyperparameters, harder to debug, likely overkill for an internship-scope project |

### Recommendation

**Approach A (compact feature-based state)** is recommended as the
baseline. It is well-suited to a from-scratch DQN built within an
internship timeframe, trains substantially faster, and is the
well-established, proven approach for classic Snake RL agents. Approach B
is noted as a possible future optimization/extension (Section 12) rather
than baseline.

This is a design recommendation only; no state-extraction code is written
in this phase.

---

## 4. Action Space

Two candidate framings exist: **absolute** (Up/Down/Left/Right) vs
**relative** (Straight/Turn Left/Turn Right).

**Recommendation: relative action space — `[Straight, Turn Left, Turn Right]`.**

Reasoning:
- With an absolute action space, the agent could select the direction
  directly opposite its current heading (e.g., moving Right and selecting
  Left), which is an instant self-collision into the snake's own neck.
  This wastes learning signal on a move that is always fatal and never
  useful, and requires extra reward-function complexity to discourage it.
- The relative action space makes illegal reversal **structurally
  impossible** — "Turn Left"/"Turn Right" are always defined relative to
  current heading, so the fourth (reverse) direction is simply not a
  representable action. This removes a whole category of wasted
  exploration without needing a penalty for it.
- It also reduces the action space from 4 to 3, slightly simplifying the
  DQN output layer and the exploration problem.

Interaction with current direction: the environment maintains the snake's
current heading internally; `Straight` keeps heading unchanged, `Turn
Left`/`Turn Right` rotate the heading 90° counter-clockwise/clockwise
respectively before computing the new head position for that step.

(Concretely: `Turn Left` rotates heading 90° counter-clockwise, `Turn Right` rotates heading 90° clockwise, `Straight` leaves heading unchanged.)

---

## 5. Reward Function

### Baseline proposal

| Event | Reward | Rationale |
|---|---|---|
| Eating food | `+10` | Strong, unambiguous positive signal tied directly to the actual objective (score). |
| Dying (wall or self collision) | `-10` | Strong negative signal to make death clearly worse than any exploratory movement. |
| Normal step (no food, no death) | `-0.01` (small constant) | A small time penalty discourages indefinite stalling/looping and encourages progress, without dominating the food/death signals. |
| Moving closer to food (Euclidean or Manhattan distance decreases) | `+0.1` | Provides a denser shaping signal so the agent gets useful gradient even before it has learned to reliably reach food. |
| Moving away from food | `-0.1` | Symmetric to the above; discourages wandering away without being punitive enough to overshadow the food/death terms. |
| Survival bonus for long episodes | *Not included in baseline* | Explicitly avoided — see reward hacking discussion below. |

### Reward hacking risks and mitigations

- **Looping/circling indefinitely**: if a survival or per-step-positive
  reward were added, the agent could learn to circle safely forever rather
  than seek food, since safe stalling would out-earn risky food-seeking.
  Mitigation: no positive per-step survival reward in baseline; the
  per-step penalty (`-0.01`) makes stalling net negative, and the episode
  step cap (Section 2) forcibly ends any indefinite loop.
- **Distance-shaping exploitation**: a purely distance-based reward could
  be exploited by oscillating toward/away from food to farm the `+0.1`
  repeatedly without ever eating. Mitigation: keep the distance-shaping
  magnitude small relative to the food reward (`+10`), so eating remains
  the dominant incentive; this will be monitored in evaluation (Section
  10) and tuned in optimization (Section 11) if oscillation is observed.
- **Death-avoidance over food-seeking**: if the death penalty is too large
  relative to food reward, the agent may learn an overly conservative
  policy that avoids all risk (including risk needed to reach food).
  Mitigation: the current 1:1 ratio (`+10` food / `-10` death) is a
  starting point to be tuned based on observed behavior.

### Tuning during optimization

All reward magnitudes above are initial hyperparameters. Section 11
defines reward-value tuning as an explicit experiment variable once
baseline training results are available.

---

## 6. DQN Architecture

| Component | Baseline choice | Notes |
|---|---|---|
| Input representation | Compact feature vector (Approach A, Section 3) | ~11 features (3 danger flags, 4 direction one-hot, 4 food-direction flags) — exact count finalized in Phase 2 |
| Input size | ~11 | Placeholder pending exact feature list finalization |
| Hidden layers | 2 fully connected layers, e.g. 256 → 256 units | A small MLP is standard and sufficient for a compact feature-based Snake state; avoids overengineering |
| Activation functions | ReLU on hidden layers | Standard, avoids vanishing-gradient issues, simple and well-understood |
| Output layer | 3 linear outputs (no activation) | One Q-value per action (Straight, Turn Left, Turn Right) |
| Loss function | Mean Squared Error (MSE) or Huber/SmoothL1 loss between predicted and target Q-values | Huber loss is a common, more stable alternative to MSE for DQN and will be considered during optimization |
| Optimizer | Adam | Standard default for DQN training; adaptive learning rate helps early-stage stability |
| Learning rate | `1e-3` (initial) | Common DQN starting point; explicitly an experimental value (Section 11) |
| Discount factor (gamma) | `0.9`–`0.99` (initial: `0.95`) | Snake episodes are moderate-length; a high-but-not-extreme gamma balances near-term (food) and longer-term (survival) planning |
| Exploration strategy | Epsilon-greedy (Section 8) | Simple, standard, well-suited to a discrete 3-action space |
| Epsilon start | `1.0` | Full exploration at the start of training, when the policy has no information yet |
| Epsilon decay | Linear or exponential decay over N episodes (exact schedule tuned in Phase 2) | Gradual shift from exploration to exploitation as the agent gains experience |
| Epsilon minimum | `0.01`–`0.05` | Keeps a small amount of exploration even late in training, to avoid the policy getting stuck |

### Why DQN (vs. plain Q-learning / vs. no NN)

- **Plain tabular Q-learning** requires an explicit table indexed by every
  discrete state. Even the compact feature-based state (Section 3) has a
  combinatorially large number of possible value combinations once food
  direction, danger flags, and heading are all considered together, and
  this grows further if the state is ever made richer. A table becomes
  memory-inefficient and slow to fill in with meaningful values.
- **DQN** replaces the table with a function approximator (neural network)
  that generalizes across similar states, so the agent doesn't need to
  visit every exact state combination to make reasonable decisions.
- A neural network is specifically useful here because it lets the same
  learned weights generalize to states not exactly seen during training
  (e.g., a new relative food/danger configuration), which a lookup table
  cannot do.

---

## 7. Experience Replay

| Aspect | Baseline decision | Justification |
|---|---|---|
| Experience contents | `(state, action, reward, next_state, done)` tuple | Standard DQN transition tuple; sufficient to compute the Bellman target |
| Replay buffer capacity | e.g., 100,000 transitions | Large enough to hold a diverse history of experience across many episodes, without unbounded memory growth |
| Batch size | e.g., 64 | Common default; balances gradient stability against compute cost per training step |
| Minimum experiences before training | e.g., 1,000–5,000 transitions | Avoids training on a tiny, highly correlated initial buffer before there is meaningful diversity |
| Sampling strategy | Uniform random sampling (baseline) | Simple and standard; prioritized replay is a possible future optimization (Section 11), not baseline |

**Why experience replay is useful for DQN:** consecutive environment steps
are highly correlated (each state is derived from the previous one), and
training directly on this sequential stream violates the i.i.d. assumption
behind stable gradient-based learning, causing oscillation or divergence.
Replay breaks this correlation by sampling randomly from a large history
buffer, and also improves data efficiency by reusing each transition in
multiple training updates rather than discarding it after one use.

---

## 8. Target Network

**Decision: yes, use a separate target network.**

| Aspect | Baseline decision |
|---|---|
| Why needed | Using the same (rapidly-changing) network to both select actions and compute Bellman targets creates a moving-target problem: the regression target shifts every update, which can destabilize training or cause divergence. |
| Update frequency | Every N steps/episodes (e.g., every 1,000 training steps), exact value tuned in Phase 2 |
| Update type | Hard update (baseline: full weight copy at the interval above) | Simpler to implement and reason about than soft (Polyak) updates; soft updates are a possible later refinement |
| Stability benefit | Keeps the Bellman target fixed between updates, giving the online network a stable regression target for a period of time, which is a well-established DQN stabilization technique |

---

## 9. Exploration vs Exploitation

- **Exploration**: taking actions that are not currently believed optimal,
  in order to discover new information about the environment.
- **Exploitation**: taking the action currently believed to maximize
  expected return, based on what has been learned so far.
- **Epsilon**: the probability of taking a random (exploratory) action
  instead of the current best-known action, in epsilon-greedy selection.
- **Epsilon decay**: the schedule by which epsilon is reduced over time,
  shifting behavior from mostly-exploration early on to mostly-exploitation
  later.

**Why too much exploration is bad:** if epsilon stays high for too long,
the agent keeps acting close to randomly and never consistently exploits
what it has learned, so training progress (and observed score/reward)
stays noisy and slow to improve.

**Why too little exploration is bad:** if epsilon decays too quickly, the
agent may lock into a suboptimal policy early (e.g., overly cautious
looping) before it has discovered better strategies, since it stops trying
alternative actions before the value estimates are reliable.

**Recommended baseline (explicitly experimental starting values):**
- Epsilon start: `1.0`
- Epsilon decay: gradual, over on the order of hundreds to low-thousands of
  episodes (exact rate to be tuned)
- Epsilon minimum: `0.01`–`0.05`

---

## 10. Training Design

| Aspect | Design |
|---|---|
| Episode structure | `reset()` → loop: select action (epsilon-greedy) → `step()` → store transition → optionally train on a replay batch → repeat until `terminated` or `truncated` |
| Maximum steps per episode | A capped value (e.g., proportional to board size) to prevent unbounded non-productive episodes |
| Action selection timing | Once per environment step, using the current online network + current epsilon |
| Experience storage timing | Immediately after each `step()` call, before any training update |
| Replay training timing | After each step (or every K steps) once the buffer has at least the minimum experience count (Section 7) |
| Target network update timing | Every N training steps/episodes (Section 8) |
| Checkpoint saving | Periodically (e.g., every M episodes) and whenever a new best evaluation score is observed |
| Best-model identification | Based on periodic evaluation-episode performance (Section 10 metrics), not training-episode reward (which includes exploration noise) |
| Random seed strategy | A fixed seed per experiment run, applied to environment reset RNG, replay sampling, and network initialization, to support reproducibility |
| Reproducibility requirements | Same seed + same config should produce comparable results across runs, modulo any unavoidable hardware-level nondeterminism |

No training loop code is written in this phase — this section defines the
intended mechanics only.

---

## 11. Performance Evaluation

| Metric | What it tells us |
|---|---|
| Average score (food eaten) | Core measure of task performance — how well the agent actually plays Snake |
| Maximum score | Best-case capability of the current policy; useful for spotting potential even if average is still low |
| Average reward | Overall reward-function performance, useful for training diagnostics, though not identical to actual "skill" |
| Survival steps (average episode length) | Indicates whether the agent has learned basic self-preservation independent of food-seeking |
| Food collected per episode | Same as score, but tracked per-episode for variance analysis |
| Episode length distribution | Reveals whether performance is consistent or highly variable across episodes |
| Performance consistency (e.g., std. dev of score across evaluation episodes) | Distinguishes a reliably good policy from one that occasionally performs well but is inconsistent |

**Evaluation protocol:**
- Evaluation must use a **greedy** policy (epsilon = 0, or a very small
  fixed epsilon), not the exploratory training policy, so that reported
  performance reflects the learned policy rather than random behavior.
- A fixed number of evaluation episodes (e.g., 20–50) run periodically
  during/after training, using a separate evaluation seed set distinct
  from training seeds where practical, to reduce the chance of the
  reported performance being an artifact of a specific seed.
- No target performance numbers are defined here — actual thresholds
  (e.g., "average score of X") will only be set after baseline training
  results exist, to avoid fabricating expectations.

---

## 12. Optimization / Experiment Plan

**Candidate tuning variables:** learning rate, gamma, epsilon decay
schedule, replay buffer size, batch size, reward values, network
architecture (layer sizes/count), maximum episode steps.

**Suggested tuning order and rationale:**

1. **Reward values first** — since reward shaping most directly determines
   *what* the agent is incentivized to learn, and reward-hacking risk
   (Section 5) means this needs early validation before deeper
   hyperparameter tuning is meaningful.
2. **Epsilon decay schedule** — exploration behavior strongly affects
   whether the agent ever discovers good strategies at all; this is cheap
   to iterate on and has an outsized early impact.
3. **Learning rate** — a standard, high-leverage DQN hyperparameter;
   tuned once the reward/exploration setup is stable, since an unstable
   reward signal makes learning-rate tuning unreliable.
4. **Gamma, replay buffer size, batch size** — secondary refinements,
   tuned after the above are reasonably stable.
5. **Network architecture size** — considered last, and only if a
   well-tuned small network still underperforms, to avoid overengineering
   before it's justified by evidence.

**Workflow:** establish a baseline run with the Section 16 configuration →
change one variable at a time → compare using the Section 10 evaluation
metrics on the same evaluation protocol → keep the change only if it
produces a measurable, reproducible improvement.

No experiments are run in this phase.

---

## 13. Testing Strategy

**Environment tests** (`tests/`, targeting `src/environment/`):
- `reset()` produces a valid starting state (correct snake length/position, valid food placement)
- Movement updates head/body position correctly for each action type
- Food spawns only in empty cells
- Eating food increases score and snake length, and spawns new food
- Snake growth correctly extends the body without dropping the tail prematurely
- Wall collision correctly triggers termination
- Self-collision correctly triggers termination
- Episode terminates on step-limit and on full-board conditions

**Agent tests** (targeting `src/agent/`):
- State input has the expected shape/type for the network
- Action output is a valid action in `{Straight, Turn Left, Turn Right}`
- Epsilon-greedy selection respects the current epsilon value (statistically, over many calls)
- Replay memory correctly stores and samples transitions, respecting capacity
- Model save/load round-trips network weights correctly

**Training tests** (targeting `src/training/`):
- A short integration test that a few training steps run without error end-to-end (environment → agent → replay → update)
- Reproducibility check: same seed produces the same initial trajectory, where practical

No test code is written in this phase — this defines what will be
implemented later.

---

## 14. Module Responsibilities

| Location | Future contents |
|---|---|
| `src/environment/` | Custom Snake environment class (reset/step/render logic), board/grid utilities, food-spawn logic |
| `src/agent/` | DQN network definition, agent class (action selection, learning update), replay buffer implementation |
| `src/training/` | Training loop/script, checkpoint-saving logic, training-time logging |
| `src/evaluation/` | Evaluation-episode runner, metrics computation, evaluation reporting |
| `src/utils/` | Shared helpers: config loading, seeding utilities, logging setup, common data structures |
| `configs/` | YAML/JSON config files defining hyperparameters (learning rate, gamma, epsilon schedule, reward values, board size, etc.) per experiment |
| `models/` | Saved model checkpoints (best model, periodic checkpoints) |
| `results/graphs/` | Training curves, score-over-time plots, evaluation comparison charts |
| `results/evaluation/` | Evaluation run outputs/reports (metrics summaries per checkpoint/experiment) |
| `tests/` | Unit/integration tests as described in Section 13 |
| `notebooks/` | Exploratory analysis, prototyping, ad hoc result inspection |
| `docs/` | This specification and future design/decision documents |

No files are created in these locations beyond this specification document
itself in this phase.

---

## 15. Architecture Diagram

```
                 ┌───────────────────┐
                 │  Snake Environment │
                 └─────────┬─────────┘
                           │  produces
                           ▼
                       ┌───────┐
                       │ State │
                       └───┬───┘
                           │  fed into
                           ▼
                    ┌──────────────┐
                    │  DQN Agent   │
                    └──────┬───────┘
                           │  selects
                           ▼
                       ┌────────┐
                       │ Action │
                       └───┬────┘
                           │  applied to
                           ▼
                 ┌───────────────────┐
                 │  Snake Environment │
                 └─────────┬─────────┘
                           │  returns
                           ▼
              ┌─────────────────────────┐
              │  Reward + Next State     │
              └────────────┬────────────┘
                           │  stored in
                           ▼
                   ┌────────────────┐
                   │ Replay Buffer  │
                   └────────┬───────┘
                           │  sampled batch
                           ▼
                   ┌────────────────┐
                   │  DQN Training  │
                   └────────┬───────┘
                           │  periodically syncs
                           ▼
                  ┌──────────────────┐
                  │  Target Network  │
                  └────────┬─────────┘
                           │  informs
                           ▼
                   ┌────────────────┐
                   │  Updated Agent │
                   └────────────────┘
```

---

## 16. Final Baseline Configuration

**(Explicitly labeled: BASELINE CONFIGURATION — initial values, subject to change during optimization.)**

| Setting | Baseline value |
|---|---|
| Game | Snake |
| Board size | 20×20 |
| State representation | Compact feature-based (Approach A, Section 3) |
| Action space | Relative: `[Straight, Turn Left, Turn Right]` |
| Reward design | Food `+10`, Death `-10`, Step `-0.01`, Toward-food `+0.1`, Away-from-food `-0.1` |
| RL algorithm | Deep Q-Network (DQN) |
| Neural network architecture | MLP: input (~11) → 256 → 256 → 3 outputs, ReLU activations |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Gamma (discount factor) | `0.95` |
| Replay buffer | Capacity 100,000; batch size 64; min experience 1,000–5,000 |
| Epsilon strategy | Start `1.0`, gradual decay, minimum `0.01`–`0.05` |
| Target network | Hard update, every N (e.g., 1,000) training steps |
| Maximum episode steps | Capped, proportional to board size |
| Evaluation metrics | Average score, max score, average reward, survival steps, food collected, performance consistency |

---

## 17. Bright Hub Requirement Alignment

| Bright Hub requirement | Our design |
|---|---|
| Select Environment | Custom Snake environment, Gymnasium-style interface (Section 2) |
| Define States | Compact feature-based state representation (Section 3) |
| Define Actions | Relative action space: Straight / Turn Left / Turn Right (Section 4) |
| Design Reward Function | Baseline reward function with hacking-risk mitigations (Section 5) |
| Model Training | DQN with experience replay and target network (Sections 6–8, 10) |
| Performance Evaluation | Defined metrics and evaluation protocol (Section 11) — no results yet |
| Optimization | Defined experiment/tuning plan and order (Section 12) — no experiments run yet |

Nothing in this document has been implemented, trained, or evaluated. All
numeric values are proposed baseline hyperparameters for Phase 2.

---

## 18. Open Decisions / Risks

- Exact compact feature list (Section 3) needs final enumeration before
  Phase 2 implementation begins (count and exact semantics of danger/food
  features).
- Exact epsilon decay schedule (linear vs exponential, and rate) is not
  yet fixed — a reasonable choice will be made in Phase 2 and revisited
  during optimization.
- Reward magnitudes (Section 5) are a first guess and are the top
  candidate for early tuning if training behavior shows reward hacking
  symptoms (e.g., circling, oscillation).
- Target network update frequency and hard-vs-soft update choice may be
  revisited if training shows instability.
- Whether to eventually explore Approach B (grid/CNN-based state) remains
  an open, lower-priority extension depending on how Approach A performs.
- Exact max-episode-steps value and its relationship to board size still
  needs a concrete formula, to be finalized in Phase 2.

