from __future__ import division
import numpy as np

from _model import GGNN
from utils_ import Data
import pickle
import argparse
import datetime
import tensorflow.compat.v1 as tf
import gymnasium as gym
# from collections import deque
import random
import seaborn as sns
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
# GNN parameters
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--num_episodes', type=int, default=3, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--step', type=int, default=1, help='gnn propagation steps')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')

# DQN parameters
parser.add_argument('--bufferSize', type=int, default=1000, help='DQN buffer size')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--tau', type=int, default=100, help='target update frequency')
parser.add_argument('--T', type=int, default=10, help='T threshold for switching training method')
parser.add_argument('--max_step', type=int, default=6, help='maximum training step of DQN')
parser.add_argument('--epsilon_max', type=float, default=1.0, help='maximum exploration rate')
parser.add_argument('--epsilon_decay', type=float, default=0.001, help='exploration decay rate')
parser.add_argument('--epsilon_min', type=float, default=0.01, help='minimum exploration decay rate')

opt = parser.parse_args()

train_data = pickle.load(open('../datasets/'+opt.dataset+'/train.txt', 'rb'))
test_data = pickle.load(open('../datasets/'+opt.dataset+'/test.txt', 'rb'))
# all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
# print('train data value')
# print(train_data)
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
print(opt)

# Set a seed for reproducibility
random.seed(42)

# Initialize z as a random variable between 0 and 1
z = random.uniform(0.0, 1.0)
epsilon = random.uniform(opt.epsilon_min, opt.epsilon_max)


class Env(gym.Env):
    def __init__(self, actor_, n_node_):
        super(Env, self).__init__()

        self.n_node = n_node_
        self.actor = actor_
        self.max_session_length = 1
        self.session_end = False
        self.current_state = []
        self.next_state = []
        self.action = []
        self.current_step = None

        self.action_space = gym.spaces.Discrete(self.n_node)  # Number of possible items to recommend
        print('action space:')
        print(self.action_space)
        self.observation_space = gym.spaces.Dict({
            'action': gym.spaces.Box(low=0, high=self.n_node-1, shape=(1,), dtype=np.int32),
            'state': gym.spaces.Box(low=-1, high=1, shape=(self.n_node-1,), dtype=np.float32)
        })

    def reset(self, max_n_n=None, seed=None, train=True):
        if train:
            self.current_state = self.actor.scores_train
        else:
            self.current_state = self.actor.scores_test
        self.current_step = 0
        return self.current_state, self.current_step

    def step(self, target_, action_, train=True):
        # if train: adj_in__, adj_out__, alias__, item__, mask__, targets__, max_n_node__ =
        # train_data.get_updated_slice(index, action_, opt.dataset) # next_state = self.actor.run(
        # self.actor.scores_train, targets__, item__, adj_in__, adj_out__, alias__, #
        # mask__, #                             None) else: adj_in__, adj_out__, alias__, item__, mask__, targets__,
        # max_n_node__ = test_data.get_updated_slice(index, action_, opt.dataset) # next_state = self.actor.run(
        # self.actor.scores_test, targets__, item__, adj_in__, adj_out__, alias__, mask__, None)
        next_step = self.current_step+1

        # Check if the session has reached the maximum length
        if next_step >= self.max_session_length:
            self.session_end = True
        else:
            self.session_end = False
        # Calculate rewards based on the shifted targets for each timestamp
        # if self.session_end:
        #     shifted_targets = targets_  # Targets for the final timestamp
        # else:
        #     shifted_targets = [sequence[next_step:next_step+self.max_session_length] for sequence in targets_]
        hit_ = []
        for _score, _target in zip(action_, target_):
            hit_.append(np.isin(_target-1, _score))
        _rewards = np.mean(hit_)
        if self.session_end:
            done_ = True
        else:
            done_ = False
        # info_ = {}
        # Update the current step for the next iteration
        self.current_step = next_step
        self.next_state = None  # with new set of parameters
        # print('next state from get_updated slice')

        return _rewards, done_

env = Env(actor, n_node)

# dqn = DQN(n_node, env, gamma=opt.gamma, batch_size=opt.batchSize, lr=opt.lr, step=opt.step,
#           decay=opt.lr_dc_step * len(train_data.inputs) / opt.batchSize,
#           lr_dc=opt.lr_dc)
# target_dqn = DQN(n_node, env, gamma=opt.gamma, batch_size=opt.batchSize, lr=opt.lr, step=opt.step,
#                  decay=opt.lr_dc_step * len(train_data.inputs) / opt.batchSize,
#                  lr_dc=opt.lr_dc)


def train_step(ggnn_actor, rec_env, _adj_in, _adj_out, _alias, _item, _mask, _targets):
    step = 0
    reward_train_step = []
    scores_ = []
    loss_actor = []
    while step <= env.max_session_length:
        reward = None
        action_ = None
        # GGNN makes recommendations for the current_state and take selected actions in the environment
        selected_actions, current_state, actor_loss, _ = ggnn_actor.run(
                [ggnn_actor.action, ggnn_actor.scores_train, ggnn_actor.loss_train, ggnn_actor.opt], _targets, _item,
                _adj_in, _adj_out, _alias, _mask, action_, reward)
        reward, _done = rec_env.step(
                _targets, selected_actions, train=True)

        _, _sac_loss = ggnn_actor.run([ggnn_actor.opt_sac, ggnn_actor.loss_sac], _targets, _item, _adj_in,
                                      _adj_out, _alias, _mask, selected_actions, reward)
        reward_train_step.append(reward)
        # if z > 0.5:
        #     if episode <= opt.T:
        #         # _dqn_loss, _, advantage = dqn_critic.run(
        #         #     [dqn_critic.loss_dqn, dqn_critic.opt_dqn, dqn_critic.advantage], current_state, _next_state, reward,
        #         #     selected_actions, actor_loss)
        #
        #         _, _sac_loss = ggnn_actor.run([ggnn_actor.opt_sac, ggnn_actor.loss_sac], _targets, _item, _adj_in,
        #                                       _adj_out, _alias, _mask, reward)
        #     else:
        #         # _dqn_loss, _, advantage = dqn_critic.run(
        #         #     [dqn_critic.loss_sac, dqn_critic.opt_sac, dqn_critic.advantage], current_state, _next_state, reward,
        #         #     selected_actions, actor_loss)
        #
        #         _, _sac_loss = ggnn_actor.run([ggnn_actor.opt_sac, ggnn_actor.loss_sac], _targets, _item, _adj_in,
        #                                       _adj_out, _alias, _mask, reward)
        #
        # else:
        #     if episode <= opt.T:
        #         # _dqn_loss_, _, advantage = dqn_critic.run(
        #         #     [dqn_critic.loss_dqn_, dqn_critic.opt_dqn_, dqn_critic.advantage], current_state, _next_state,
        #         #     reward,
        #         #     selected_actions, actor_loss)
        #
        #         _, _sac_loss = ggnn_actor.run([ggnn_actor.opt_sac, ggnn_actor.loss_sac], _targets, _item, _adj_in,
        #                                       _adj_out, _alias, _mask, reward)
        #     else:
        #         # _dqn_loss, _, advantage = dqn_critic.run(
        #         #     [dqn_critic.loss_sac_, dqn_critic.opt_sac_, dqn_critic.advantage],
        #         #     current_state, _next_state, reward, selected_actions, actor_loss)
        #
        #         _, _sac_loss = ggnn_actor.run([ggnn_actor.opt_sac, ggnn_actor.loss_sac], _targets, _item, _adj_in,
        #                                       _adj_out, _alias, _mask, reward)

        # _alias, _item, _adj_in, _adj_out, _mask, _targets = alias_, item_, adj_in_, adj_out_, mask_, targets_
        step += 1
        scores_.append(current_state)
        loss_actor.append(actor_loss)
    return reward_train_step, loss_actor, scores_


rewards_for_each_episode_ = []
rewards_for_each_episode = []
best_result = [0, 0]
best_epoch = [0, 0]
for episode in range(opt.num_episodes):
    slices = train_data.generate_batch(actor.batch_size)
    # print('start training: ', datetime.datetime.now())
    rewards_for_each_slice = []
    train_loss = []
    # Initialize an index to keep track of sessions within the slice
    for i, j in zip(slices, np.arange(len(slices))):
        adj_in, adj_out, alias, item, mask, targets, max_n_node = train_data.get_slice(i, opt.dataset)
        # Training Loop
        state, _ = env.reset()
        # print('state = env.reset()')
        # print(state)
        advantage = None
        action = None
        episode_reward = 0.0
        rewards_in_10_step = []

        rewards, score_train, loss_train_ = train_step(actor, env, adj_in, adj_out, alias, item, mask,
                                                       targets)
        #     print(f'reward for 10 steps', np.mean(rewards))
        rewards_for_each_slice.append(np.mean(rewards))
        train_loss.append(loss_train_)
    loss=np.mean(train_loss)
    # print('rewards_for_each_slice')
    # print(np.mean(rewards_for_each_slice))
    rewards_for_each_episode_.append(np.mean(rewards_for_each_slice))

    slices_ = test_data.generate_batch(actor.batch_size)
    print('start testing: ', datetime.datetime.now())
    rewards_for_each_slice_ = []
    hit, mrr, test_loss_ = [], [], []
    # Initialize an index to keep track of sessions within the slice
    for i, j in zip(slices_, np.arange(len(slices_))):
        adj_in, adj_out, alias, item, mask, targets, max_n_node = test_data.get_slice(i, opt.dataset)
        rewards_in_10_step_ = []
        state, _ = env.reset()
        advantage = None
        action = None
        scores, test_loss = actor.run([actor.scores_test, actor.loss_test], targets, item, adj_in, adj_out, alias, mask, action, advantage)
        test_loss_.append(test_loss)
        index = np.argsort(scores, 1)[:, -20:]
        for score, target in zip(index, targets):
            hit.append(np.isin(target-1, score))
            if len(np.where(score == target-1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (20-np.where(score == target-1)[0][0]))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    test_loss = np.mean(test_loss_)
    if hit >= best_result[0]:
        best_result[0] = hit
        best_epoch[0] = episode
    if mrr >= best_result[1]:
        best_result[1] = mrr
        best_epoch[1] = episode
    print('train_loss:\t%.4f\ttest_loss:\t%4f\tPrecision@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' %
          (loss, test_loss, best_result[0], best_result[1], best_epoch[0], best_epoch[1]))



    # print('rewards_for_each_slice')
    # print(np.mean(rewards_for_each_slice_))
#     rewards_for_each_episode_.append(np.mean(rewards_for_each_slice_))
# print('rewards_for_each_episode_')
# print(np.mean(rewards_for_each_episode_))
#
#

# plt.figure(figsize=(8, 6))
# plt.plot(opt.num_episodes, rewards_for_each_episode_, marker='o', linestyle='-', color='b', label='Hit Rate')
# plt.title('30 epochs, 5time steps, Actor-Critic model')
# plt.xlabel('Epoch')
# plt.ylabel('Hit Rate')
# plt.grid(True)
# plt.legend(loc='best')
#
# plt.tight_layout()
# plt.show()
# plot model's performance
# hit = rewards_for_each_episode_
# epoch = np.arange(opt.num_episodes)
# plt.title('Actor-Critic model 30.T.5')
# plt.xlabel('Epoch')
# plt.ylabel('Hit Rate')
# sns.scatterplot(x=epoch, y=hit)
# plt.show()
