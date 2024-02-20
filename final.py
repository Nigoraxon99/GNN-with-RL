from __future__ import division
import numpy as np

from model import GGNN
from utility import Data
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
parser.add_argument('--dataset', default='yoochoose1_64', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--num_episodes', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--step', type=int, default=1, help='gnn propagation steps')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')

# DQN parameters
parser.add_argument('--max_step', type=int, default=6, help='maximum training step of DQN')

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

# rewards_for_each_episode_ = []
# rewards_for_each_episode = []
best_result = [0, 0]
best_epoch = [0, 0]
for episode in range(opt.num_episodes):
    slices = train_data.generate_batch(actor.batch_size)
    print('start training: ', datetime.datetime.now())
    # rewards_for_each_slice = []
    train_loss = []
    # Initialize an index to keep track of sessions within the slice
    for i, j in zip(slices, np.arange(len(slices))):
        adj_in, adj_out, alias, item, mask, targets = train_data.get_slice(i)
        # Training Loop
        scores_train, train_loss_, _, _, reward, rm, last_id, last_h, seq_h, last, seq, m, coef, b, ma, y1, action, act_prob, _ = actor.run([actor.scores_train, actor.loss_sac, actor.opt, actor.opt_sac, actor.reward, actor.rm, actor.last_id, actor.last_h, actor.seq_h, actor.last, actor.seq, actor.m, actor.coef, actor.b, actor.ma, actor.y1, actor.action, actor.act_prob, actor.global_step], targets, item, adj_in, adj_out,
                                            alias, mask)
        #     print(f'reward for 10 steps', np.mean(rewards))
        # rewards_for_each_slice.append(np.mean(reward))
        train_loss.append(train_loss_)

    loss = np.mean(train_loss)
    # print('rewards_for_each_slice')
    # print(np.mean(rewards_for_each_slice))
    # rewards_for_each_episode_.append(np.mean(rewards_for_each_slice))

    slices_ = test_data.generate_batch(actor.batch_size)
    print('start testing: ', datetime.datetime.now())
    # rewards_for_each_slice_ = []
    hit, mrr, test_loss_ = [], [], []
    # Initialize an index to keep track of sessions within the slice
    for i, j in zip(slices_, np.arange(len(slices_))):
        adj_in, adj_out, alias, item, mask, targets = test_data.get_slice(i)
        # print(f'adj_in', adj_in)
        # print(f'adj_out', adj_out)
        # print(f'alias', alias)
        # print(f'item', item)
        # print(f'targets', targets)
        # rewards_in_10_step_ = []
        scores, test_loss, reward_test = actor.run([actor.scores_test, actor.loss_test, actor.reward], targets, item, adj_in, adj_out, alias, mask)
        test_loss_.append(test_loss)
        # print(f'logits', scores)
        # exp_x = np.exp(scores-np.max(scores, axis=-1, keepdims=True))
        # action_probabilities = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        # print(f'action_probabilities', action_probabilities)
        index = np.argsort(scores, 1)[:, -20:]
        for score, target in zip(index, targets):
            hit.append(np.isin(target-1, score))
            if len(np.where(score == target-1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (20-np.where(score == target-1)[0][0]))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    test_loss__ = np.mean(test_loss_)
    if hit >= best_result[0]:
        best_result[0] = hit
        best_epoch[0] = episode
    if mrr >= best_result[1]:
        best_result[1] = mrr
        best_epoch[1] = episode
    print('train_loss:\t%.4f\ttest_loss:\t%4f\tPrecision@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' %
          (loss, test_loss__, best_result[0], best_result[1], best_epoch[0], best_epoch[1]))

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
