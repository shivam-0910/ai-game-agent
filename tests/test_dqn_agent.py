"""Test suite for the DQN agent, network, and replay buffer (Phase 3).

Covers network shape/behavior, replay buffer capacity/sampling, agent
initialization, epsilon-greedy action selection, the learning step,
target-network updates, save/load persistence, and an environment
integration smoke test.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from src.agent.dqn_agent import DQNAgent
from src.agent.dqn_network import DQNNetwork, INPUT_SIZE, OUTPUT_SIZE
from src.agent.replay_buffer import ReplayBuffer
from src.environment.snake_env import SnakeEnv


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def rng_state():
    torch.manual_seed(0)


@pytest.fixture
def small_agent() -> DQNAgent:
    """An agent configured with a tiny replay/min-experience threshold
    so learning tests run fast without needing thousands of steps."""
    return DQNAgent(
        min_experiences=8,
        batch_size=4,
        replay_capacity=200,
        target_update_frequency=3,
        seed=0,
    )


def _random_state() -> np.ndarray:
    return np.random.default_rng(0).random(INPUT_SIZE).astype(np.float32)


def _fill_buffer(agent: DQNAgent, count: int) -> None:
    rng = np.random.default_rng(1)
    for i in range(count):
        state = rng.random(INPUT_SIZE).astype(np.float32)
        next_state = rng.random(INPUT_SIZE).astype(np.float32)
        action = i % agent.action_size
        reward = float(rng.uniform(-1, 1))
        done = i % 5 == 0
        agent.remember(state, action, reward, next_state, done)


# ---------------------------------------------------------------------
# Network tests
# ---------------------------------------------------------------------


class TestDQNNetwork:
    def test_can_instantiate(self):
        net = DQNNetwork()
        assert net is not None

    def test_single_input_shape_accepted(self):
        net = DQNNetwork()
        state = torch.rand(INPUT_SIZE)
        output = net(state)
        assert output.shape == (OUTPUT_SIZE,)

    def test_output_size_is_three(self):
        net = DQNNetwork()
        state = torch.rand(1, INPUT_SIZE)
        output = net(state)
        assert output.shape[-1] == 3

    def test_output_is_finite(self):
        net = DQNNetwork()
        state = torch.rand(1, INPUT_SIZE)
        output = net(state)
        assert torch.isfinite(output).all()

    def test_batch_input_works(self):
        net = DQNNetwork()
        batch = torch.rand(16, INPUT_SIZE)
        output = net(batch)
        assert output.shape == (16, OUTPUT_SIZE)

    def test_parameters_exist_and_are_trainable(self):
        net = DQNNetwork()
        params = list(net.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_no_softmax_outputs_can_be_negative_or_exceed_one(self):
        """Confirms the output layer is unconstrained (no softmax),
        since Q-values are not probabilities."""
        net = DQNNetwork()
        # Force weights so at least one output is clearly outside [0, 1].
        with torch.no_grad():
            last_layer = net.network[-1]
            last_layer.bias.fill_(5.0)
        output = net(torch.zeros(1, INPUT_SIZE))
        assert (output > 1.0).any()


# ---------------------------------------------------------------------
# Replay buffer tests
# ---------------------------------------------------------------------


class TestReplayBuffer:
    def test_starts_empty(self):
        buffer = ReplayBuffer(capacity=10)
        assert len(buffer) == 0

    def test_add_increases_size(self):
        buffer = ReplayBuffer(capacity=10)
        buffer.add(_random_state(), 0, 1.0, _random_state(), False)
        assert len(buffer) == 1

    def test_size_increases_correctly_over_multiple_adds(self):
        buffer = ReplayBuffer(capacity=10)
        for i in range(5):
            buffer.add(_random_state(), 0, 1.0, _random_state(), False)
        assert len(buffer) == 5

    def test_capacity_is_respected(self):
        buffer = ReplayBuffer(capacity=5)
        for i in range(20):
            buffer.add(_random_state(), 0, 1.0, _random_state(), False)
        assert len(buffer) == 5

    def test_old_experiences_discarded_when_full(self):
        buffer = ReplayBuffer(capacity=3)
        markers = [np.full(INPUT_SIZE, i, dtype=np.float32) for i in range(5)]
        for marker in markers:
            buffer.add(marker, 0, 0.0, marker, False)
        stored_first_values = {exp.state[0] for exp in buffer._buffer}
        # Only the 3 most recent markers (2, 3, 4) should remain.
        assert stored_first_values == {2.0, 3.0, 4.0}

    def test_sampling_returns_correct_batch_size(self):
        buffer = ReplayBuffer(capacity=50)
        for _ in range(20):
            buffer.add(_random_state(), 0, 1.0, _random_state(), False)
        batch = buffer.sample(batch_size=8)
        assert batch.states.shape[0] == 8
        assert batch.actions.shape[0] == 8
        assert batch.rewards.shape[0] == 8
        assert batch.next_states.shape[0] == 8
        assert batch.dones.shape[0] == 8

    def test_can_sample_false_when_insufficient(self):
        buffer = ReplayBuffer(capacity=50)
        buffer.add(_random_state(), 0, 1.0, _random_state(), False)
        assert buffer.can_sample(10) is False

    def test_sample_raises_when_insufficient(self):
        buffer = ReplayBuffer(capacity=50)
        buffer.add(_random_state(), 0, 1.0, _random_state(), False)
        with pytest.raises(ValueError):
            buffer.sample(batch_size=10)

    def test_sampled_data_shapes_and_types(self):
        buffer = ReplayBuffer(capacity=50)
        for i in range(20):
            buffer.add(_random_state(), i % 3, 0.5, _random_state(), i % 4 == 0)
        batch = buffer.sample(batch_size=8)
        assert batch.states.shape == (8, INPUT_SIZE)
        assert batch.next_states.shape == (8, INPUT_SIZE)
        assert batch.states.dtype == np.float32
        assert batch.actions.dtype == np.int64
        assert batch.rewards.dtype == np.float32
        assert batch.dones.dtype == np.float32


# ---------------------------------------------------------------------
# Agent initialization tests
# ---------------------------------------------------------------------


class TestAgentInitialization:
    def test_agent_initializes(self):
        agent = DQNAgent()
        assert agent is not None

    def test_policy_and_target_networks_exist(self):
        agent = DQNAgent()
        assert isinstance(agent.policy_network, DQNNetwork)
        assert isinstance(agent.target_network, DQNNetwork)

    def test_policy_and_target_initially_match(self):
        agent = DQNAgent(seed=0)
        for p_param, t_param in zip(
            agent.policy_network.parameters(), agent.target_network.parameters()
        ):
            assert torch.equal(p_param, t_param)

    def test_optimizer_exists(self):
        agent = DQNAgent()
        assert agent.optimizer is not None

    def test_replay_buffer_exists(self):
        agent = DQNAgent()
        assert isinstance(agent.replay_buffer, ReplayBuffer)

    def test_epsilon_initialized_to_start_value(self):
        agent = DQNAgent(epsilon_start=0.7)
        assert agent.epsilon == 0.7

    def test_gamma_configured_correctly(self):
        agent = DQNAgent(gamma=0.9)
        assert agent.gamma == 0.9

    def test_device_selection_works(self):
        agent = DQNAgent()
        assert agent.device.type in ("cpu", "cuda")

    def test_rejects_invalid_gamma(self):
        with pytest.raises(ValueError):
            DQNAgent(gamma=1.5)

    def test_rejects_invalid_epsilon_bounds(self):
        with pytest.raises(ValueError):
            DQNAgent(epsilon_start=0.1, epsilon_min=0.5)


# ---------------------------------------------------------------------
# Action selection tests
# ---------------------------------------------------------------------


class TestActionSelection:
    def test_action_always_valid(self):
        agent = DQNAgent(seed=0)
        state = _random_state()
        for _ in range(50):
            action = agent.select_action(state)
            assert action in (0, 1, 2)

    def test_epsilon_zero_is_greedy_and_deterministic(self):
        agent = DQNAgent(epsilon_start=0.0, epsilon_min=0.0, seed=0)
        state = _random_state()
        actions = {agent.select_action(state) for _ in range(20)}
        # Greedy selection from a fixed network on a fixed state must
        # always return the same action.
        assert len(actions) == 1

    def test_epsilon_zero_matches_argmax(self):
        agent = DQNAgent(epsilon_start=0.0, epsilon_min=0.0, seed=0)
        state = _random_state()
        action = agent.select_action(state)
        state_tensor = torch.as_tensor(state).unsqueeze(0)
        with torch.no_grad():
            expected = int(torch.argmax(agent.policy_network(state_tensor), dim=1).item())
        assert action == expected

    def test_epsilon_one_produces_varied_actions_over_many_calls(self):
        agent = DQNAgent(epsilon_start=1.0, epsilon_min=1.0, seed=0)
        state = _random_state()
        actions = {agent.select_action(state) for _ in range(200)}
        # With full exploration over 200 draws from 3 actions, we should
        # see more than one distinct action (astronomically unlikely not
        # to, avoiding a flaky single-outcome assertion).
        assert len(actions) > 1

    def test_training_false_ignores_epsilon(self):
        agent = DQNAgent(epsilon_start=1.0, epsilon_min=1.0, seed=0)
        state = _random_state()
        actions = {agent.select_action(state, training=False) for _ in range(20)}
        assert len(actions) == 1

    def test_epsilon_decay_reduces_epsilon(self):
        agent = DQNAgent(epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.9)
        agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.9)

    def test_epsilon_decay_respects_minimum(self):
        agent = DQNAgent(epsilon_start=0.02, epsilon_min=0.01, epsilon_decay=0.5)
        for _ in range(20):
            agent.decay_epsilon()
        assert agent.epsilon >= 0.01
        assert agent.epsilon == pytest.approx(0.01)


# ---------------------------------------------------------------------
# Learning tests
# ---------------------------------------------------------------------


class TestLearning:
    def test_can_store_experiences(self, small_agent: DQNAgent):
        state = _random_state()
        small_agent.remember(state, 0, 1.0, state, False)
        assert len(small_agent.replay_buffer) == 1

    def test_cannot_learn_before_minimum_replay_size(self, small_agent: DQNAgent):
        _fill_buffer(small_agent, count=3)  # below min_experiences=8
        assert small_agent.can_learn() is False
        with pytest.raises(RuntimeError):
            small_agent.learn()

    def test_can_learn_after_enough_experiences(self, small_agent: DQNAgent):
        _fill_buffer(small_agent, count=20)
        assert small_agent.can_learn() is True
        loss = small_agent.learn()
        assert isinstance(loss, float)

    def test_loss_is_finite(self, small_agent: DQNAgent):
        _fill_buffer(small_agent, count=20)
        loss = small_agent.learn()
        assert np.isfinite(loss)

    def test_parameters_change_after_learning(self, small_agent: DQNAgent):
        _fill_buffer(small_agent, count=20)
        before = [p.clone() for p in small_agent.policy_network.parameters()]
        small_agent.learn()
        after = list(small_agent.policy_network.parameters())
        changed = any(
            not torch.equal(b, a) for b, a in zip(before, after)
        )
        assert changed

    def test_target_network_unchanged_by_single_learn_call(self, small_agent: DQNAgent):
        # target_update_frequency=3 in the fixture, so a single learn()
        # call (1st step) must not yet trigger a target update.
        _fill_buffer(small_agent, count=20)
        target_before = copy.deepcopy(small_agent.target_network.state_dict())
        small_agent.learn()
        target_after = small_agent.target_network.state_dict()
        for key in target_before:
            assert torch.equal(target_before[key], target_after[key])

    def test_target_network_updates_at_configured_frequency(self, small_agent: DQNAgent):
        _fill_buffer(small_agent, count=20)
        for _ in range(small_agent.target_update_frequency):
            small_agent.learn()
        # After exactly target_update_frequency learn() calls, target
        # should now match the (just-updated) policy network.
        for p_param, t_param in zip(
            small_agent.policy_network.parameters(), small_agent.target_network.parameters()
        ):
            assert torch.equal(p_param, t_param)

    def test_manual_target_update(self, small_agent: DQNAgent):
        _fill_buffer(small_agent, count=20)
        small_agent.learn()  # policy network now differs from target
        small_agent.update_target_network()
        for p_param, t_param in zip(
            small_agent.policy_network.parameters(), small_agent.target_network.parameters()
        ):
            assert torch.equal(p_param, t_param)

    def test_target_network_is_separate_object(self, small_agent: DQNAgent):
        assert small_agent.policy_network is not small_agent.target_network


# ---------------------------------------------------------------------
# Save / load tests
# ---------------------------------------------------------------------


class TestSaveLoad:
    def test_save_creates_file(self, tmp_path):
        agent = DQNAgent(seed=0)
        checkpoint_path = tmp_path / "checkpoint.pt"
        agent.save(checkpoint_path)
        assert checkpoint_path.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        agent = DQNAgent(seed=0)
        checkpoint_path = tmp_path / "nested" / "dir" / "checkpoint.pt"
        agent.save(checkpoint_path)
        assert checkpoint_path.exists()

    def test_load_restores_equivalent_output(self, tmp_path):
        agent = DQNAgent(seed=0)
        state = _random_state()
        state_tensor = torch.as_tensor(state).unsqueeze(0)
        with torch.no_grad():
            original_output = agent.policy_network(state_tensor).clone()

        checkpoint_path = tmp_path / "checkpoint.pt"
        agent.save(checkpoint_path)

        new_agent = DQNAgent(seed=1)  # different init, should differ before load
        with torch.no_grad():
            before_load_output = new_agent.policy_network(state_tensor)
        assert not torch.allclose(original_output, before_load_output)

        new_agent.load(checkpoint_path)
        with torch.no_grad():
            loaded_output = new_agent.policy_network(state_tensor)
        assert torch.allclose(original_output, loaded_output)

    def test_load_restores_epsilon(self, tmp_path):
        agent = DQNAgent(seed=0, epsilon_start=1.0, epsilon_decay=0.5)
        agent.decay_epsilon()
        agent.decay_epsilon()
        saved_epsilon = agent.epsilon

        checkpoint_path = tmp_path / "checkpoint.pt"
        agent.save(checkpoint_path)

        new_agent = DQNAgent(seed=1)
        new_agent.load(checkpoint_path)
        assert new_agent.epsilon == pytest.approx(saved_epsilon)

    def test_load_restores_target_network(self, tmp_path):
        agent = DQNAgent(seed=0, min_experiences=8, batch_size=4)
        _fill_buffer(agent, count=20)
        agent.learn()  # policy now differs from target's original state

        checkpoint_path = tmp_path / "checkpoint.pt"
        agent.save(checkpoint_path)

        new_agent = DQNAgent(seed=1)
        new_agent.load(checkpoint_path)
        for p_param, t_param in zip(
            agent.target_network.parameters(), new_agent.target_network.parameters()
        ):
            assert torch.equal(p_param, t_param)


# ---------------------------------------------------------------------
# Environment integration smoke test (not training)
# ---------------------------------------------------------------------


class TestEnvironmentIntegration:
    def test_agent_and_env_interfaces_are_compatible(self):
        env = SnakeEnv(board_size=10)
        agent = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=int(env.action_space.n),
            min_experiences=5,
            batch_size=4,
            seed=0,
        )

        state, _info = env.reset(seed=0)
        assert state.shape == (agent.state_size,)

        for _ in range(30):
            action = agent.select_action(state)
            assert env.action_space.contains(action)

            next_state, reward, terminated, truncated, _info = env.step(action)
            agent.remember(state, action, reward, next_state, terminated or truncated)
            state = next_state

            if agent.can_learn():
                loss = agent.learn()
                assert np.isfinite(loss)

            if terminated or truncated:
                state, _info = env.reset()

        # No exceptions raised across the full observation -> agent ->
        # action -> env.step() -> agent.remember() -> agent.learn() loop
        # confirms interface compatibility end-to-end.
