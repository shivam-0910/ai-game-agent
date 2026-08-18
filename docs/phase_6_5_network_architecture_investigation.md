# Phase 6.5 Network Architecture Investigation

## Current Architecture
- Structure: 11 (state features) → 256 → 256 → 3 (Q-values)
- Activations: ReLU between hidden layers, linear output
- Fixed in: `src/agent/dqn_network.py` (HIDDEN_SIZE = 256)

## Framework Support
- **Configurable via ExperimentConfig**: YES
- `ExperimentConfig.network_hidden_size` field exists (line 92-94)
- `run_experiment.py` handles architecture swaps (lines 132-153, 218-253)
- **No source code changes required**

## Proposed Screening Experiments

| ID | hidden_size | Rationale |
|---|---|---|
| A0 (control) | 256 | Current baseline |
| A1 | 128 | Smaller capacity (half baseline) - tests if model is over-parameterized |
| A2 | 512 | Larger capacity (double baseline) - tests if model capacity-limited |

## Fixed Parameters (Phase 6.1-6.4 adopted)
- REWARD_DEATH = -5.0
- epsilon_decay = 0.995
- learning_rate = 0.001
- gamma = 0.95
- replay_capacity = 100,000
- batch_size = 64
- min_experiences = 1,000
- target_update_frequency = 1,000

## Methodology
- Screening: 100 training episodes, 100 evaluation episodes
- Validation: 500 training episodes for promising candidate
- Seeds: training=42, evaluation_base=1,000,000
- Decision rule: validation score must exceed Phase 6.1 reference (34.13)

## Validation Candidate
- Select the screening candidate with highest average score
- If validation > 34.13, adopt; otherwise reject
- If no candidate > control in screening, reject all
