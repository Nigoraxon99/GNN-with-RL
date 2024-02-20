from __future__ import division
import numpy as np

from model import GGNN
from utils_ import Data
import pickle
import argparse
import datetime
import tensorflow.compat.v1 as tf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from typing import Union, Dict
from collections import deque
import random

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
print('train data value')
print(train_data)
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
    def __init__(self, actor_, n_node_):
        super(Env, self).__init__()

        self.train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
        self.test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))

        self.n_node = n_node_
        self.actor = actor_
        self.max_session_length = 30
        self.embedding_dim = 100

        self.action_space = gym.spaces.Discrete(self.n_node)  # Number of possible items to recommend
        print('action space:')
        print(self.action_space)
        # Define your custom observation space here

        item_indices_low = 0
        item_indices_high = self.n_node  # max_n_node
        self.observation_space = gym.spaces.Box(
            low=item_indices_low,
            high=item_indices_high,
            shape=(self.actor.batch_size, self.max_session_length),
            dtype=np.int32
        )
        self.session_end = False
        self.current_state = []
        self.next_state = []
        self.action = []
        self.current_step = None

    def reset(self, max_n_n=None, seed=None):

        self.current_state = self.actor.item
        if max_n_n < self.max_session_length:

            # Pad the item matrix with zeros
            zero_padding = tf.zeros((self.actor.batch_size, self.max_session_length - max_n_node), dtype=tf.int32)

            # Concatenate the zero padding to the end of the state tensor
            padded_state = tf.concat([self.current_state, zero_padding], axis=1)

            # Use padded_state as self.current_state
            self.current_state = padded_state
        else:
            self.current_state = self.current_state
        self.current_state = actor.run(tf.cast(self.current_state, tf.float32),
                                       targets, item, adj_in, adj_out,
                                       alias,
                                       mask)
        self.current_step = 0
        current_state_numpy = np.array(self.current_state)

        # # Create feature names for the rows
        # feature_names = [f'feature{i}' for i in range(1, 101)]  # You mentioned you have 100 features
        #
        # # Convert the current_state matrix into a dictionary with feature names
        # observation_dict = {feature_name: current_state_numpy[i, :] for i, feature_name in enumerate(feature_names)}
        #
        # def get_observation() -> Union[np.ndarray, Dict[str, np.ndarray]]:
        #     return observation_dict
        #
        # # Call the function to get the observation data
        # observation = get_observation()
        print('current state:')
        print(current_state_numpy)
        return current_state_numpy, self.current_step

    def step(self, action, next_item=None):

        action_cast = tf.cast(action, tf.int32)
        action_ = tf.expand_dims(action_cast, axis=-1)
        self.action = action_
        # Remove the last zero element
        self.current_state = self.current_state[:, :-1]
        # Concatenate the action to the end
        next_state_ = tf.concat([self.current_state, self.action], -1)
        print('next state after taking actions:')
        print(next_state_)
        next_step = self.current_step + 1

        # Check if the session has reached the maximum length
        if next_step >= self.max_session_length:
            self.session_end = True
        else:
            self.session_end = False

        # Calculate the reward for the action
        rewards = self.calculate_reward(self.action, next_item)

        if self.session_end:
            done_ = True
        else:
            done_ = False
        info_ = {}
        # Update the current step for the next iteration
        self.current_step = next_step
        next_state_ = actor.run(tf.cast(next_state_, tf.float32),
                                targets, item, adj_in, adj_out,
                                alias,
                                mask)
        self.next_state = np.array(next_state_)

        return self.next_state, rewards, done_, info_, {}

    def calculate_reward(self, recommended_items, next_item_):

        # Assuming 'next_item' and 'recommended_items' are TensorFlow tensors
        next_item_in_session = tf.cast(next_item_, tf.int32)
        recommended_items = tf.cast(recommended_items, tf.int32)

        # Check if the selected items match the next item in the session
        correct_selections = tf.equal(recommended_items, next_item_in_session)
        correct_selections = tf.cast(correct_selections, tf.float32)
        # Calculate the reward as 1 for correct selections and 0 for incorrect selections
        reward_ = tf.reduce_mean(correct_selections)
        reward_np = self.actor.run(reward_,
                                   targets, item, adj_in, adj_out,
                                   alias,
                                   mask)
        return reward_np


class EnvWrapper(gym.Env):
    def __init__(self, env_, max_n_n, next_item):
        self.env = env_
        self.max_n_node = max_n_n
        self.next_item = next_item
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, max_n_n=None, seed=None):
        if max_n_n is None:
            max_n_n = self.max_n_node
        return self.env.reset(max_n_n=max_n_n)

    def step(self, action, next_item=None):
        if next_item is None:
            next_item = self.next_item
        return self.env.step(action, next_item=next_item)


# Create a DQN model
env = Env(actor, n_node)

# Training Loop
# Actor (GNN) makes recommendations
slices = train_data.generate_batch(actor.batch_size)
print('start training: ', datetime.datetime.now())
# Initialize an index to keep track of sessions within the slice
for i, j in zip(slices, np.arange(len(slices))):
    adj_in, adj_out, alias, item, mask, targets, max_n_node = train_data.get_slice(i)
    wrapped_env = EnvWrapper(env, max_n_node, targets)
    critic = DQN("MlpPolicy", wrapped_env, verbose=1, **dqn_params)
    for episode in range(opt.num_episodes):
        state, _ = wrapped_env.reset()
        print('state = env.reset()')
        print(state)
        total_reward = 0
        done = False
        while not done:
            _, loss, _, scores_train = actor.run(
                [actor.opt, actor.loss_train, actor.global_step, actor.scores_train],
                targets, item, adj_in, adj_out,
                alias,
                mask)
            # Apply softmax to logit to get a probability distribution
            action_probabilities = tf.nn.softmax(scores_train)
            print('action_probabilities tensor by gnn actor:')
            print(action_probabilities)
            print('action_probabilities by gnn actor:')
            action_probabilities_np = actor.run(action_probabilities, targets, item, adj_in, adj_out, alias,
                                                mask)
            print(action_probabilities_np)
            print('item matrix by gnn actor:')
            print(item)

            # Get the index of the item with the highest score (the highest probability)
            selected_actions = tf.argmax(action_probabilities,
                                         axis=1)  # list of selected item index for each session
            # next_train_data = train_data.update_data(j, selected_actions)
            selected_actions_np = actor.run(selected_actions, targets, item, adj_in, adj_out, alias,
                                            mask)
            print('selected actions by gnn actor:')
            print(selected_actions)
            print('selected actions by gnn actor:')
            print(selected_actions_np)

            # Interact with the environment to get rewards and next state
            next_state, reward, done, info, _ = wrapped_env.step(selected_actions)

            # Store the experience in a replay buffer
            # replay_buffer.add((state, action, reward, next_state, done))

            print('next state tensor after concatenating selected actions')
            print(next_state)

            print('reward of taking selected actions by gnn actor:')
            print(reward)

            q_values = critic.predict(state)
            q_values_next = critic.predict(next_state)

            print('q_values for current_state')
            print(q_values)
            print('q_values for next_state')
            print(q_values_next)
            # Compute target Q-values using the Bellman equation
            # target_q_values = reward + opt.gamma * q_values

            critic.learn(total_timesteps=10000, log_interval=4)
            critic.save("critic_model")

            del critic  # remove to demonstrate saving and loading

            critic = DQN.load("critic_model")

            # advantage = q_values - q_values_next
            #
            # actor.update(scores_train, advantage)

            total_reward += reward

            # Check if the episode is done
            if done:
                break
            else:
                state = next_state
