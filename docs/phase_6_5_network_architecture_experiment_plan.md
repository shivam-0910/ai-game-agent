# Phase 6.5 Network Architecture Experiment Plan

## Objective
Determine whether changing the DQN hidden-layer size improves Snake agent performance.

## Screening Matrix

| ID | network_hidden_size | Description |
|---|---|---|
| A0 (control) | 256 | Current baseline architecture |
| A1 | 128 | Reduced capacity (half baseline) |
| A2 | 512 | Increased capacity (double baseline) |

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
- Training seed: 42
- Evaluation base seed: 1,000,000
- Model selection: best-training-score checkpointing
- Only network_hidden_size varies between experiments

## Validation Rule
Promising screening candidate → 500-episode validation
Validation must exceed Phase 6.1 reference score (34.13) to be adopted
