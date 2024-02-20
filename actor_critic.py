from __future__ import division
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from model import GGNN
from utils_ import Data
import pickle
import argparse
import datetime
import tensorflow.compat.v1 as tf
from gym import spaces
from stable_baselines3 import DQN
from collections import deque
import random

import collections
import gym
import statistics
import tqdm
from typing import Any, List, Sequence, Tuple

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

parser = argparse.ArgumentParser()
# GNN parameters
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--num_episodes', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--step', type=int, default=1, help='gnn propagation steps')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')

parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
opt = parser.parse_args()

train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
# all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))

# Number of nodes is equal to the number of items
if opt.dataset == 'diginetica':
    n_node = 43098
elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
    n_node = 37484
else:
    n_node = 310
# g = build_graph(all_train_seq)
train_data = Data(train_data, sub_graph=True, shuffle=True)
test_data = Data(test_data, sub_graph=True, shuffle=False)
# Instantiate actor model
actor = GGNN(hidden_size=opt.hiddenSize, out_size=opt.hiddenSize, batch_size=opt.batchSize, n_node=n_node,
             lr=opt.lr, l2=opt.l2, step=opt.step, decay=opt.lr_dc_step * len(train_data.inputs) / opt.batchSize,
             lr_dc=opt.lr_dc)

# # Instantiate critic model
# critic = DQN(input_size=actor.out_size, output_size=len(np.unique(train_data.inputs)), learning_rate=0.001,
#              discount_factor=0.99)
print(opt)

# Define DQN hyperparameters
dqn_params = {
    "learning_rate": 0.001,
    "gamma": 0.99,
    "exploration_final_eps": 0.1,
    "exploration_fraction": 0.2,
    "buffer_size": 10000,
    "batch_size": 64,
    "train_freq": 1,
    "target_update_interval": 100,
    "learning_starts": 100,
    "max_grad_norm": 10,
}


class Env(gym.Env):
    def __init__(self, train_data, test_data, actor, n_node):
        super(Env, self).__init__()

        self.train_data = train_data
        self.test_data = test_data
        self.n_node = n_node
        self.actor = actor

        self.action_space = spaces.Discrete(self.n_node)  # Number of possible items to recommend
        print('action space:')
        print(self.action_space)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.actor.out_size,))
        self.embedding_dim = 100

        self.max_session_length = 30
        self.session_end = False
        self.current_state = []
        self.current_step = None

    def reset(self, seed=None):

        # Get the state representation from the GNN model
        self.current_state = self.actor.item
        state_dqn = actor.run(self.current_state, targets, item, adj_in, adj_out, alias, mask)
        self.current_step = 0
        return state_dqn, self.current_state

    def step(self, action):

        action_ = tf.cast(tf.expand_dims(action, axis=-1), tf.int32)

        # Insert the selected actions into the current state
        next_state_ = tf.concat([self.current_state, action_], axis=-1)

        next_step = self.current_step + 1

        # Check if the session has reached the maximum length
        if next_step >= self.max_session_length:
            self.session_end = True
        else:
            self.session_end = False

        # Calculate the reward for the action
        reward_ = self.calculate_reward(action_)

        if self.session_end:
            done_ = True
        else:
            done_ = False
        info = {}
        # Update the current state and step for the next iteration
        self.current_state = next_state_
        self.current_step = next_step

        return next_state_, reward_, done_, info

    def calculate_reward(self, recommended_items):

        # Get the next item in the session (the item that should come next)
        next_item_in_session = self.current_state[:, self.current_step + 1, :]

        # Check if the selected items match the next item in the session
        correct_selections = tf.equal(recommended_items, next_item_in_session)

        # Calculate the reward as 1 for correct selections and 0 for incorrect selections
        reward_ = tf.reduce_mean(tf.cast(correct_selections, tf.float32))

        return reward_


# Instantiate the environment
env = Env(train_data, test_data, actor, n_node)

# Create a DQN model
critic = DQN("MlpPolicy", env, verbose=1, **dqn_params)


# Wrap Gym's `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.

@tf.numpy_function(Tout=[tf.float32, tf.int32, tf.int32])
def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns state, reward and done flag given an action."""

    state, reward, done, truncated, info = env.step(action)
    return (state.astype(np.float32),
            np.array(reward, np.int32),
            np.array(done, np.int32))


slices = train_data.generate_batch(actor.batch_size)
for i, j in zip(slices, np.arange(len(slices))):
    adj_in, adj_out, alias, item, mask, targets = train_data.get_slice(i)


def run_episode(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)
        # Actor (GNN) makes recommendations
        _, action_logits_t = actor.run([actor.loss_train, actor.scores_train], targets, item, adj_in, adj_out, alias,
                                       mask)
        value = critic.predict(state)
        # Sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)
        # Apply softmax to logit to get a probability distribution
        # Store critic values
        values = values.write(t, tf.squeeze(value))

        # Store log probability of the action chosen
        action_probs = action_probs.write(t, action_probs_t[0, action])

        # Apply action to the environment to get next state and reward
        state, reward, done = env_step(action)
        state.set_shape(initial_state_shape)

        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards


def get_expected_return(
        rewards: tf.Tensor,
        gamma: float,
        standardize: bool = True) -> tf.Tensor:
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) /
                   (tf.math.reduce_std(returns) + eps))

    return returns


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


def compute_loss(
        action_probs: tf.Tensor,
        values: tf.Tensor,
        returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined Actor-Critic loss."""

    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


@tf.function
def train_step(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        gamma: float,
        max_steps_per_episode: int) -> tf.Tensor:
    """Runs a model training step."""

    with tf.GradientTape() as tape:
        # Run the model for one episode to collect training data
        action_probs, values, rewards = run_episode(
            initial_state, model, max_steps_per_episode)

        # Calculate the expected returns
        returns = get_expected_return(rewards, gamma)

        # Convert training data to appropriate TF tensor shapes
        action_probs, values, returns = [
            tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

        # Calculate the loss values to update our network
        loss = compute_loss(action_probs, values, returns)

    # Compute the gradients from the loss
    grads_actor = tape.gradient(loss, actor.vars)
    grads_critic = tape.gradient(loss, critic.trainable_variables)

    # Apply the gradients to the actor's parameters
    optimizer.apply_gradients(zip(grads_actor, actor.vars))
    optimizer.apply_gradients(zip(grads_critic, critic.trainable_variables))

    # Apply the gradients to the critic's parameters

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward


min_episodes_criterion = 100
max_episodes = 10000
max_steps_per_episode = 500

# `CartPole-v1` is considered solved if average reward is >= 475 over 500
# consecutive trials
reward_threshold = 475
running_reward = 0

# The discount factor for future rewards
gamma = 0.99

# Keep the last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

t = tqdm.trange(max_episodes)
for i in t:
    initial_state_dqn, initial_state = env.reset()
    initial_state = tf.constant(initial_state, dtype=tf.float32)
    episode_reward = int(train_step(
        initial_state, actor, critic, optimizer, gamma, max_steps_per_episode))

    episodes_reward.append(episode_reward)
    running_reward = statistics.mean(episodes_reward)

    t.set_postfix(
        episode_reward=episode_reward, running_reward=running_reward)

    # Show the average episode reward every 10 episodes
    if i % 10 == 0:
        pass  # print(f'Episode {i}: average reward: {avg_reward}')

    if running_reward > reward_threshold and i >= min_episodes_criterion:
        break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')


