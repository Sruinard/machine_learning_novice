import dataclasses
import random
from collections import deque
from typing import NamedTuple
import os

import gym
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jaxlib
import numpy as np
import optax
import sonnet as snt
import tensorflow as tf
import tree
from jax.experimental import jax2tf

ENV = gym.make('LunarLander-v2', new_step_api=True)


@dataclasses.dataclass
class Experience:
    state: jnp.ndarray
    action: int
    reward: float
    next_state: jnp.ndarray
    done: bool


class TrainConfig(NamedTuple):
    NUM_FEATURES_PER_STATE = 8
    NUM_ACTIONS = 4
    MEMORY_SIZE = 10000
    BATCH_SIZE = 32
    UPDATE_PARAMS_EVERY_N_STEPS = 4
    TAU = 0.001
    E_MIN = 0.01
    E_DECAY = 0.995
    N_EPISODES = 2000
    MAX_N_STEPS_PER_EPISODE = 1000
    SAVE_DIR = "./lunar/dqn/"


class TrainingState(NamedTuple):
    params: hk.Params
    target_params: hk.Params
    eval_params: hk.Params
    opt_state: optax.OptState


class Batch(NamedTuple):
    states: jnp.ndarray
    actions: int
    rewards: float
    next_states: jnp.ndarray
    dones: bool


def network_fn(x: jnp.ndarray) -> jnp.ndarray:
    model = hk.Sequential(
        [
            hk.Linear(64), jax.nn.relu,
            hk.Linear(64), jax.nn.relu,
            hk.Linear(TrainConfig.NUM_ACTIONS),
        ]
    )
    return model(x)


def get_random_batch(memory):
    batch = random.sample(memory, k=TrainConfig.BATCH_SIZE)
    return Batch(
        states=jnp.array([e.state for e in batch]),
        actions=jnp.array([e.action for e in batch]),
        rewards=jnp.array([e.reward for e in batch]),
        next_states=jnp.array([e.next_state for e in batch]),
        dones=jnp.array([e.done for e in batch]),
    )


def update_epsilon(epsilon, train_config: TrainConfig):
    return max(train_config.E_MIN, train_config.E_DECAY*epsilon)


def exploit_or_explore(q_value: jnp.ndarray, epsilon: float = 0.1) -> int:
    """Exploit or explore according to epsilon-greedy policy."""
    if random.random() < epsilon:
        return ENV.action_space.sample()
    else:
        return np.array(jnp.argmax(q_value))


def is_update_params(n_steps_taken: int, train_config: TrainConfig) -> bool:
    """Update params every `update_params_every` steps."""
    return (n_steps_taken + 1) % train_config.UPDATE_PARAMS_EVERY_N_STEPS == 0


def init_network_and_opt(network, optimizer, n_features):
    random_batch_size = 32
    batch = jrandom.normal(jrandom.PRNGKey(0), (random_batch_size, n_features))
    initial_params = network.init(jrandom.PRNGKey(0), batch)
    initial_opt_state = optimizer.init(initial_params)
    return initial_params, initial_opt_state


def create_variable(path, value):
    name = '/'.join(map(str, path)).replace('~', '_')
    return tf.Variable(value, name=name)


class JaxModule(snt.Module):
    def __init__(self, params, apply_fn, polymorphic_shapes, name=None):
        super().__init__(name=name)
        self._params = tree.map_structure_with_path(create_variable, params)
        self._apply = jax2tf.convert(lambda p, x: apply_fn(
            p, x), polymorphic_shapes=polymorphic_shapes)
        self._apply = tf.autograph.experimental.do_not_convert(self._apply)

    def __call__(self, inputs):
        return self._apply(self._params, inputs)


def save_network(network, train_state: TrainingState, train_config: TrainConfig):
    polymorphic_state_shape = [None, jax2tf.shape_poly.PolyShape(
        "b", train_config.NUM_FEATURES_PER_STATE
    )]

    net = JaxModule(train_state.params,
                    network.apply, polymorphic_state_shape)

    @tf.function(autograph=False, input_signature=[tf.TensorSpec([None, train_config.NUM_FEATURES_PER_STATE])])
    def forward(x):
        return net(x)

    to_save = tf.Module()
    to_save.forward = forward
    to_save.params = list(net.variables)
    if not os.path.exists(train_config.SAVE_DIR):
        os.makedirs(train_config.SAVE_DIR)
    tf.saved_model.save(to_save, train_config.SAVE_DIR)


def train(env: gym.Env, train_state: TrainingState, train_config: TrainConfig, optimizer, network, target_network):
    memory = deque(maxlen=train_config.MEMORY_SIZE)

    @jax.jit
    def loss(params, target_params, batch):
        q_values = network.apply(params, batch.states)
        q_values_pred = q_values[jnp.arange(q_values.shape[0]), batch.actions]

        q_values_next = target_network.apply(target_params, batch.next_states)
        q_values_next_max = jnp.max(q_values_next, axis=1)

        q_value_true = batch.rewards + \
            jnp.where(batch.dones, 0.0, q_values_next_max)
        return jnp.mean((q_values_pred - q_value_true) ** 2)

    @jax.jit
    def update(train_state: TrainingState, batch: Batch) -> TrainingState:
        """Learning rule (stochastic gradient descent)."""
        grads = jax.grad(loss)(train_state.params,
                               train_state.target_params, batch)
        updates, opt_state = optimizer.update(
            grads, train_state.opt_state)
        params = optax.apply_updates(train_state.params, updates)

        # Update target network.
        # params * TAU + (1 - TAU) * new_params
        # target_params = params * TrainConfig.TAU  + (1 - TrainConfig.TAU) * train_state.target_params
        target_params = optax.incremental_update(
            params, train_state.target_params, TrainConfig.TAU)

        # Compute avg_params, the exponential moving average of the "live" params.
        # We use this only for evaluation (cf. https://doi.org/10.1137/0330046).
        eval_params = optax.incremental_update(
            params, train_state.eval_params, step_size=0.001)
        return TrainingState(params, target_params, eval_params, opt_state)

    total_reward_history = []
    moving_average_window_size = 100
    epsilon = 1.0

    for episode in range(train_config.N_EPISODES):
        state = env.reset()
        total_reward = 0.0

        for step in range(train_config.MAX_N_STEPS_PER_EPISODE):
            q_value = network.apply(train_state.params, state)
            action = exploit_or_explore(q_value=q_value, epsilon=epsilon)

            next_state, reward, is_done, *_ = env.step(action)
            experience = Experience(state, action, reward, next_state, is_done)
            memory.append(experience)
            if len(memory) < train_config.MEMORY_SIZE:
                state = next_state
                total_reward += reward
                if is_done:
                    break
                continue

            if is_update_params(step, train_config=train_config):
                batch = get_random_batch(memory)
                train_state = update(train_state, batch)

            state = next_state
            total_reward += reward
            if is_done:
                break

        total_reward_history.append(total_reward)
        mean_total_reward_in_window = np.mean(
            total_reward_history[-moving_average_window_size:])
        epsilon = update_epsilon(epsilon, train_config)

        print(f"\rEpisode {episode+1} | Total point average of the last {moving_average_window_size} episodes: {mean_total_reward_in_window:.2f}", end="")

        if (episode+1) % moving_average_window_size == 0:
            print(
                f"\rEpisode {episode+1} | Total point average of the last {moving_average_window_size} episodes: {mean_total_reward_in_window:.2f}")
        # We will consider that the environment is solved if we get an
        # average of 200 points in the last 100 episodes.
        if mean_total_reward_in_window >= 200.0:
            print(f"\n\nEnvironment solved in {episode+1} episodes!")
            # q_network.save('lunar_lander_model.h5')
            save_network(network, train_state, train_config)
            break


def main():
    train_config = TrainConfig()
    network = hk.without_apply_rng(hk.transform(network_fn))
    target_network = hk.without_apply_rng(hk.transform(network_fn))
    optimizer = optax.adam(1e-3)
    # Initialise network and optimiser; note we draw an input to get shapes.
    initial_params, initial_opt_state = init_network_and_opt(
        network, optimizer, train_config.NUM_FEATURES_PER_STATE)
    train_state = TrainingState(
        initial_params, initial_params, initial_params, initial_opt_state)
    train(env=ENV, train_state=train_state, train_config=train_config,
          optimizer=optimizer, network=network, target_network=target_network)


if __name__ == "__main__":
    main()
