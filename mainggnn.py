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

        self.action_space = spaces.Discrete(self.n_node)  # Number of possible items to recommend
        print('action space:')
        print(self.action_space)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.embedding_dim,), dtype=np.float32)

        self.session_end = False
        self.current_state = []
        self.observation = []
        self.current_step = None

    def reset(self, seed=None):

        # Get the state representation
        self.current_state = self.train_data
        # state_ = np.array(self.current_state)
        print('current state:')
        print(self.current_state)
        self.current_step = 0

        return self.current_state

    def step(self, action, next_item, updated_data):

        # action_ = tf.expand_dims((tf.cast(action, tf.int32)), axis=1)
        # next_state_ = tf.concat([self.current_state, action_], -1)
        print('next state after taking actions:')
        print(updated_data)
        next_step = self.current_step + 1

        # Check if the session has reached the maximum length
        if next_step >= self.max_session_length:
            self.session_end = True
        else:
            self.session_end = False

        # Calculate the reward for the action
        reward_ = self.calculate_reward(action, next_item)

        if self.session_end:
            done_ = True
        else:
            done_ = False
        info_ = {}
        # Update the current step for the next iteration
        self.current_step = next_step

        return updated_data, reward_, done_, info_, self.current_step

    def calculate_reward(self, recommended_items, next_item):

        # Assuming 'next_item' and 'recommended_items' are NumPy arrays
        next_item_in_session = next_item.astype(np.int32)

        # Check if the selected items match the next item in the session
        correct_selections = np.equal(recommended_items, next_item_in_session)

        # Calculate the reward as 1 for correct selections and 0 for incorrect selections
        reward_ = np.mean(correct_selections.astype(np.float32))

        return reward_


# Instantiate the environment
env = Env(actor, n_node)

# env = DummyVecEnv([lambda: rec_env])

# Create a DQN model
critic = DQN("MlpPolicy", env, verbose=1, **dqn_params)

# Training Loop
for episode in range(opt.num_episodes):
    state = env.reset()
    print('state = env.reset()')
    print(state)
    total_reward = 0
    done = False

    while not done:
        train_data = Data(state, sub_graph=True, shuffle=True)
        # Actor (GNN) makes recommendations
        slices = train_data.generate_batch(actor.batch_size)
        print('start training: ', datetime.datetime.now())
        # Initialize an index to keep track of sessions within the slice
        for i, j in zip(slices, np.arange(len(slices))):
            adj_in, adj_out, alias, item, mask, targets = train_data.get_slice(i)

            _, loss, _, scores_train = actor.run([actor.opt, actor.loss_train, actor.global_step, actor.scores_train],
                                                 targets, item, adj_in, adj_out,
                                                 alias,
                                                 mask)
            # Apply softmax to logit to get a probability distribution
            action_probabilities = tf.nn.softmax(scores_train)
            print('action_probabilities by gnn actor:')
            print(action_probabilities)
            print('item matrix by gnn actor:')
            print(item)

            # Get the index of the item with the highest score (the highest probability)
            selected_actions = tf.argmax(action_probabilities, axis=1)  # list of selected item index for each session
            selected_actions = tf.expand_dims((tf.cast(selected_actions, tf.int32)), axis=1)
            # next_train_data = train_data.update_data(j, selected_actions)
            selected_actions_np = actor.run(selected_actions, targets, item, adj_in, adj_out, alias,
                                            mask)
            print('selected actions by gnn actor:')
            print(selected_actions)
            print('selected actions value by gnn actor:')
            print(selected_actions_np)
            updated_train_data = []
            # Convert selected_actions to a Python list (assuming selected_actions_np is a NumPy array)
            selected_actions_list = selected_actions_np.tolist()
            print('selected actions list by gnn actor:')
            print(selected_actions_list)

            # Get the selected actions for this slice
            selected_actions_slice = selected_actions_list[j]

            for session_idx, action_idx in enumerate(selected_actions_slice):
                # Map the action indices to the session and items within the session
                session_index = slices[session_idx] + i[0]  # Adjust the session index
                selected_item = action_idx

                # Update the corresponding session in tra_seqs with the selected action
                state[0][session_index].append(selected_item)

            # state = updated_state

            print('resulting sessions(updated train_data)')
            print(state)
            # Interact with the environment to get rewards and next state
            next_state, reward, done, info, _ = env.step(selected_actions_np, targets, state)

            # Store the experience in a replay buffer
            # replay_buffer.add((state, action, reward, next_state, done))

            # action_probabilities_np, state_value, next_state_value = actor.run(
            #     [action_probabilities, state, next_state], targets, item, adj_in, adj_out,
            #     alias,
            #     mask)

            # print('action_probabilities value by gnn actor:')
            # print(action_probabilities_np)

            print('current state by env.reset()')
            print(state)
            # print('current state value by env.reset()')
            # print(state_value)
            print('next state tensor after concatenating selected actions')
            print(next_state)
            # print('next state tensor value after concatenating selected actions')
            # print(next_state_value)
            print('reward of taking selected actions by gnn actor:')
            print(reward)

            _action, _states = critic.predict(state, deterministic=True)
            action_next_, states_next_ = critic.predict(next_state)
            # print('q_values for current_state')
            # print(q_values)
            # print('q_values for next_state')
            # print(q_values_next)

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
