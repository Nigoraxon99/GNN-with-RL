from __future__ import division
import numpy as np
from model import GGNN
from stable_baselines3 import DQN
from utils_ import build_graph, Data, split_validation
# from stable_baselines3.common.vec_env import DummyVecEnv
import pickle
import argparse
import datetime
import tensorflow.compat.v1 as tf
import gym
from gym import spaces

parser = argparse.ArgumentParser()
# GNN parameters
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--num_episodes', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
# DQN parameters
parser.add_argument('--dqn_actions', type=int, default=10, help='Number of DQN actions')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor gamma')
parser.add_argument('--dqn_lr', type=float, default=0.001, help='DQN learning rate')
parser.add_argument('--total_timesteps', type=int, default=10, help='total_timesteps')
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
print('train_data.inputs')
print(train_data.inputs)
unique_items = np.unique(train_data.inputs)
print('length of unique_items')
print(len(unique_items))
# Instantiate actor model
actor = GGNN(hidden_size=opt.hiddenSize, out_size=opt.hiddenSize, batch_size=opt.batchSize, n_node=n_node,
             lr=opt.lr, l2=opt.l2, step=opt.step, decay=opt.lr_dc_step * len(train_data.inputs) / opt.batchSize,
             lr_dc=opt.lr_dc)

slices = train_data.generate_batch(actor.batch_size)
fetches = [actor.opt, actor.loss_train, actor.global_step]
print('start training: ', datetime.datetime.now())
loss_ = []
for i, j in zip(slices, np.arange(len(slices))):
    adj_in, adj_out, alias, item, mask, targets = train_data.get_slice(i)
    actor.run(fetches, targets, item, adj_in, adj_out, alias, mask)

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


class RecommenderEnv(gym.Env):
    def __init__(self, train_data, test_data, actor, n_node):
        super(RecommenderEnv, self).__init__()

        # Initialize your train and test data here
        self.train_data = train_data
        self.test_data = test_data
        self.n_node = n_node
        self.actor = actor
        self.embedding_dim = 100
        self.max_episode_length = 100  # Define the maximum episode length

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(unique_items))  # Number of possible items to recommend
        self.observation_space = spaces.Box(low=-1, high=1, shape=(actor.out_size,))

        # Initialize `current_session` as an empty array
        self.current_session = []
        self.current_step = None

    def reset(self):

        # Get the state representation from the GNN model
        self.current_session = self.actor.ggnn()  # Extract the last time step as state
        print('self.current_session')
        print(self.current_session)
        self.current_step = 0

        # Return the initial state, which is typically the session history
        # initial_state = self.sampled_session.get_history()
        return self.current_session

    def step(self, sampled_actions_np):

        # Use the DQN critic to estimate Q-values for each action
        q_values = critic.predict(sampled_actions_np)

        print("q_values:")
        print(q_values)

        # Assuming q_values is a tuple with the NumPy array as the first element
        q_values_array = q_values[0]

        # Print the value of the NumPy array
        print('q_values np array')
        print(q_values_array)

        # Find the action with the highest Q-value
        best_action_index = np.argmax(q_values_array)

        # The 'best_action_index' now contains the index of the action with the highest Q-value
        # Map this index back to the original action space to get the recommended action
        recommended_action = sampled_actions[best_action_index]

        # Print the value of the NumPy array
        print('recommended_action')
        print(recommended_action)

        # take reommended_action in the environment: Append the recommended item to the session
        self.current_session.append(recommended_action)

        # Update the current step
        self.current_step += 1

        # Check if the episode is done (end of session)
        done = self.is_episode_done()

        # Calculate immediate reward based on your criteria (e.g., match with the actual next item)
        immediate_reward = self.calculate_immediate_reward(recommended_action)

        obs = self.current_session

        # Return the new state, immediate reward, done status, and any additional info
        return obs, immediate_reward, done, {}

    def calculate_immediate_reward(self, recommended_action):
        # Get the actual next item in the session based on the current step
        actual_next_item = self.train_data.inputs[self.current_step]

        # Compare the recommended item with the actual next item
        if recommended_action == actual_next_item:
            immediate_reward = 1  # Reward of 1 if the recommendation is correct
        else:
            immediate_reward = 0  # Reward of 0 if the recommendation is incorrect

        return immediate_reward

    def is_episode_done(self):
        # the episode is done if the current step exceeds the session length
        return self.current_step >= len(self.train_data.inputs) - 1


# Instantiate the environment
env = RecommenderEnv(train_data, test_data, actor, n_node)
# env = DummyVecEnv([lambda: env])

# Access the observation space shape
print('observation space shape:')
print(env.observation_space.shape)

print('action space shape:')
print(env.action_space.shape)

# Create a DQN model
critic = DQN("MlpPolicy", env, verbose=1, **dqn_params)
critic.learn(total_timesteps=10000, log_interval=4)
critic.save("critic_model")

del critic  # remove to demonstrate saving and loading

critic = DQN.load("critic_model")
dqn_optimizer = tf.keras.optimizers.Adam(learning_rate=opt.dqn_lr)

for episode in range(opt.num_episodes):

    # Reset the environment for a new episode
    state_dqn, state = env.reset()
    done = False
    episode_reward = 0

    with tf.GradientTape() as tape:
        while not done:
            # Use the GNN actor to generate an action vector

            # Use the GNN actor model to get action logits
            _, logits = actor.forward(state, train=False)  # Set train=False for inference

            # Apply softmax to logits to get a probability distribution
            action_probabilities = tf.nn.softmax(logits)
            print("action_probabilities:")
            print(action_probabilities)

            # Sample actions from the categorical distribution defined by action_probabilities
            sampled_actions = tf.squeeze(tf.multinomial(tf.log(action_probabilities), num_samples=1), axis=1)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                action_probabilities_np = sess.run(action_probabilities,
                                                   feed_dict={actor.adj_in: adj_in, actor.adj_out: adj_out,
                                                              actor.alias: alias, actor.item: item, actor.mask: mask,
                                                              actor.tar: targets})
                sampled_actions_np = sess.run(sampled_actions, feed_dict={actor.adj_in: adj_in, actor.adj_out: adj_out,
                                                                          actor.alias: alias, actor.item: item,
                                                                          actor.mask: mask, actor.tar: targets})
            print("state:")
            print(state)
            print("action_probabilities_np:")
            print(action_probabilities_np)
            print("sampled_actions_np:")
            print(sampled_actions_np)

            # Reshape sampled_actions to have the same shape as action_probabilities_np
            sampled_actions = np.tile(sampled_actions_np, (action_probabilities_np.shape[1], 1))

            sampled_actions = np.transpose(sampled_actions)

            # Calculate the action_vector using element-wise multiplication
            action_vector = np.sum(action_probabilities_np * sampled_actions,
                                   axis=1)  # Use axis=1 to sum across actions

            # Convert the resulting action_vector back to a TensorFlow tensor if needed
            action_vector = tf.convert_to_tensor(action_vector, dtype=tf.float32)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                action_vector_np = sess.run(action_vector,
                                            feed_dict={actor.adj_in: adj_in, actor.adj_out: adj_out, actor.alias: alias,
                                                       actor.item: item, actor.mask: mask, actor.tar: targets})

            print("action_vector:")
            print(action_vector)

            print("action_vector_np:")
            print(action_vector_np)

            # Select the action with the highest Q-value
            # action = tf.argmax(q_values_array)

            # Execute the action in the environment
            next_state, reward, done, _ = env.step(sampled_actions_np)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                next_state_np = sess.run(next_state,
                                         feed_dict={actor.adj_in: adj_in, actor.adj_out: adj_out, actor.alias: alias,
                                                    actor.item: item, actor.mask: mask, actor.tar: targets})

            # Execute the action and observe the next state, reward, and done
            # next_state, reward, done, _ = env.step(action_np)

            print("next state")
            print(next_state)

            print("next state np")
            print(next_state_np)

            # Update the DQN critic using the Bellman equation
            target_q_values = reward + opt.gamma * critic.predict(next_state_np)
            loss = tf.reduce_mean(tf.square(q_values - target_q_values))

            # Compute gradients and apply updates
            dqn_gradients = tape.gradient(loss, critic.trainable_variables)
            dqn_optimizer.apply_gradients(zip(dqn_gradients, critic.trainable_variables))

            episode_reward += reward

            # Update the GNN actor using the estimated Q-value (policy gradient)
            actor_loss = -tf.reduce_mean(tf.math.log(action_probabilities) * q_values)
            actor_gradients = tape.gradient(actor_loss, actor.trainable_variables)
            actor.optimizer.apply_gradients(zip(actor_gradients, actor.trainable_variables))

            # Update the current state
            state = next_state

    # Print episode results
    print(f"Episode {episode}: Total Reward = {episode_reward}")
