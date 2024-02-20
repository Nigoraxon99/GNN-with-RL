# import gym
# from gym import spaces
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tensorflow.compat.v1 as tf

session = tf.Session()

class RecommenderEnv(gym.Env):
    def __init__(self, train_data, test_data, actor, n_node):
        super(RecommenderEnv, self).__init__()

        # Initialize your train and test data here
        self.train_data = train_data
        self.test_data = test_data
        self.n_node = n_node
        self.actor = actor

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(np.unique(train_data.inputs)))  # Number of possible items to recommend
        print('action space:')
        print(self.action_space)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.actor.out_size,))
        self.embedding_dim = 100

        # Initialize other variables like episode length, current session, etc.
        self.max_episode_length = 100  # Define the maximum episode length
        # Initialize `current_session` as an empty array
        self.current_session = []
        self.current_step = None

    def reset(self, seed=None):

        # Get the state representation from the GNN model
        self.current_session = self.actor.ggnn()
        # Extract the desired tensor shape ([self.out_size, ]) from self.current_session
        # critic_current_state = self.actor.get_fin_state()

        with session.as_default():
            session.run(tf.global_variables_initializer())
            current_state_np = self.current_session.eval(feed_dict)
        # session.close()
        print('self.current_session')
        print(self.current_session)
        self.current_step = 0
        current_session_dqn_np = np.mean(current_state_np, axis=(0, 1))
        print('current_session_dqn_np shape')
        print(current_session_dqn_np)

        return current_session_dqn_np, self.current_session

    def step(self, action):

        # print("q_values:")
        # print(q_values)
        #
        # # Assuming q_values is a tuple with the NumPy array as the first element
        # q_values_array = q_values[0]
        #
        # # Print the value of the NumPy array
        # print('q_values np array')
        # print(q_values_array)

        # Find the action with the highest Q-value
        best_action_index = np.argmax(q_values_array)

        # The 'best_action_index' now contains the index of the action with the highest Q-value
        # Map this index back to the original action space to get the recommended action
        rec_act = action[best_action_index]

        # Print the value of the NumPy array
        print('recommended_action')
        print(rec_act)

        # take recommended_action in the environment: Append the recommended item to the session
        self.current_session.append(rec_act)

        # Update the current step
        self.current_step += 1

        # Check if the episode is done (end of session)
        done = self.is_episode_done()

        # Calculate immediate reward based on your criteria (e.g., match with the actual next item)
        immediate_reward = self.calculate_immediate_reward(rec_act)

        print('next session(next_state in our case)')
        print(self.current_session)

        # Return the new state, immediate reward, done status, and any additional info
        return self.current_session, immediate_reward, done, {}

    def calculate_immediate_reward(self, rec_action):
        # Get the actual next item in the session based on the current step
        actual_next_item = self.train_data.inputs[self.current_step]

        # Compare the recommended item with the actual next item
        if rec_action == actual_next_item:
            immediate_reward = 1  # Reward of 1 if the recommendation is correct
        else:
            immediate_reward = 0  # Reward of 0 if the recommendation is incorrect

        return immediate_reward

    def is_episode_done(self):
        # the episode is done if the current step exceeds the session length
        return self.current_step >= len(self.train_data.inputs) - 1

