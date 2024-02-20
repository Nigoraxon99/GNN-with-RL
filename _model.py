import tensorflow.compat.v1 as tf
import math

tf.disable_v2_behavior()


class Model(object):
    def __init__(self, hidden_size=100, out_size=100, batch_size=100, n_node=None, lr=None):
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.lr = lr
        self.mask = tf.placeholder(dtype=tf.float32)
        self.alias = tf.placeholder(dtype=tf.int32)
        self.item = tf.placeholder(dtype=tf.int32)
        self.tar = tf.placeholder(dtype=tf.int32)
        self.action = tf.placeholder(dtype=tf.int32)
        self.reward = tf.placeholder(dtype=tf.float32)

        self.stdv = 1.0 / math.sqrt(self.hidden_size)

        self.nasr_w1 = tf.get_variable('nasr_w1', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w2 = tf.get_variable('nasr_w2', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_v = tf.get_variable('nasrv', [1, self.out_size], dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_b = tf.get_variable('nasr_b', [self.out_size], dtype=tf.float32, initializer=tf.zeros_initializer())

    def forward(self, re_embedding, train=True):
        rm = tf.reduce_sum(self.mask, 1)
        last_id = tf.gather_nd(self.alias, tf.stack([tf.range(self.batch_size), tf.to_int32(rm)-1], axis=1))
        last_h = tf.gather_nd(re_embedding, tf.stack([tf.range(self.batch_size), last_id], axis=1))
        seq_h = tf.stack([tf.nn.embedding_lookup(re_embedding[i], self.alias[i]) for i in range(self.batch_size)],
                         axis=0)  # batch_size*T*d
        last = tf.matmul(last_h, self.nasr_w1)
        seq = tf.matmul(tf.reshape(seq_h, [-1, self.out_size]), self.nasr_w2)
        last = tf.reshape(last, [self.batch_size, 1, -1])
        m = tf.nn.sigmoid(last+tf.reshape(seq, [self.batch_size, -1, self.out_size])+self.nasr_b)
        coef = tf.matmul(tf.reshape(m, [-1, self.out_size]), self.nasr_v, transpose_b=True) * tf.reshape(
            self.mask, [-1, 1])
        b = self.embedding[1:]

        ma = tf.concat([tf.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, 1),
                        tf.reshape(last, [-1, self.out_size])], -1)
        self.B = tf.get_variable('B', [2 * self.out_size, self.out_size],
                                 initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        y1 = tf.matmul(ma, self.B)
        logits = tf.matmul(y1, b, transpose_b=True)
        self.vars = tf.trainable_variables()
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tar-1, logits=logits))
        if train:
            lossL2 = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in self.vars if
                               v.name not in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.L2
            loss = loss+lossL2

        # ∇J(θ) = E[∇log(π(a|s;θ)*A(s,a))]  REINFORCE
        # Calculate the action probabilities of selected_actions
        action_probabilities = tf.nn.softmax(logits)
        # selected_actions = tf.argmax(action_probabilities, axis=1)  # a
        # # Cast selected_actions to the appropriate data type (int32)
        # action = tf.cast(selected_actions, tf.int32)
        # reward = tf.reduce_mean(tf.cast(tf.equal(self.tar - 1, action), dtype=tf.float32))
        # # print(f'reward', reward)
        # # Create indices for the selected actions
        # indices = tf.range(0, tf.shape(action)[0])  # Create indices for each batch
        # Gather the selected action probabilities
        selected_action_probabilities = tf.gather(action_probabilities, self.action, axis=1)  # π(a|s;θ)
        log_probabilities = tf.math.log(selected_action_probabilities)
        actor_loss = -tf.reduce_mean(log_probabilities * self.reward)
        if train:
            entropy_term = -tf.reduce_mean(selected_action_probabilities * log_probabilities)
            actor_loss += self.lr * entropy_term
        loss_sac = actor_loss+loss
        return loss, loss_sac, logits, self.reward

    def run(self, fetches, tar, item, adj_in, adj_out, alias, mask, action, reward):
        return self.sess.run(fetches, feed_dict={self.tar: tar, self.item: item, self.adj_in: adj_in,
                                                 self.adj_out: adj_out, self.alias: alias, self.mask: mask, self.action: action, self.reward: reward})


class GGNN(Model):
    def __init__(self, hidden_size=100, out_size=100, batch_size=300, n_node=None,
                 lr=None, l2=None, step=1, decay=None, lr_dc=0.1):
        super(GGNN, self).__init__(hidden_size, out_size, batch_size, n_node, lr)
        self.state = None
        self.embedding = tf.get_variable(shape=[n_node, hidden_size], name='embedding', dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.adj_in = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.adj_out = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.n_node = n_node
        self.lr = lr
        self.L2 = l2
        self.step = step
        self.W_in = tf.get_variable('W_in', shape=[self.out_size, self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in = tf.get_variable('b_in', [self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out = tf.get_variable('W_out', [self.out_size, self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out = tf.get_variable('b_out', [self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        with tf.variable_scope('actor', reuse=None):
            self.loss_train, self.loss_sac, self.scores_train, self.reward = self.forward(self.ggnn())
        with tf.variable_scope('actor', reuse=True):
            self.loss_test, self.loss_sac, self.scores_test, self.reward = self.forward(self.ggnn(), train=False)
        self.global_step = tf.Variable(0)
        self.learning_rate = tf.train.exponential_decay(lr, global_step=self.global_step, decay_steps=decay,
                                                        decay_rate=lr_dc, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.opt = self.optimizer.minimize(self.loss_train, global_step=self.global_step)
        self.opt_sac = self.optimizer.minimize(self.loss_sac, global_step=self.global_step)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        print('session')
        print(self.sess)

    def ggnn(self):
        fin_state = tf.nn.embedding_lookup(self.embedding, self.item)
        cell = tf.nn.rnn_cell.GRUCell(self.out_size, reuse=tf.compat.v1.AUTO_REUSE, name='gru')
        with tf.variable_scope('gru'):
            for i in range(self.step):
                fin_state = tf.reshape(fin_state, [self.batch_size, -1, self.out_size])
                fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                    self.W_in)+self.b_in, [self.batch_size, -1, self.out_size])
                fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                     self.W_out)+self.b_out, [self.batch_size, -1, self.out_size])
                av = tf.concat([tf.matmul(self.adj_in, fin_state_in),
                                tf.matmul(self.adj_out, fin_state_out)], axis=-1)
                state_output, fin_state = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(av, [-1, 2 * self.out_size]), axis=1),
                                      initial_state=tf.reshape(fin_state, [-1, self.out_size]))
        return tf.reshape(fin_state, [self.batch_size, -1, self.out_size])

#
# class DQN(tf.keras.Model):
#     def __init__(self, num_actions, env, gamma=0.99, batch_size=100, lr=None, step=1, decay=None, lr_dc=0.1):
#         super(DQN, self).__init__()
#         self.batch_size = batch_size
#         self.num_actions = num_actions
#         self.gamma = gamma
#         self.actor_loss = tf.placeholder(dtype=tf.float32)
#         self.state = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.num_actions-1])
#         self.next_state = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.num_actions-1])
#         self.action = tf.placeholder(dtype=tf.int32, shape=self.batch_size)
#         self.reward = tf.placeholder(dtype=tf.float32)
#         self.env = env
#         self.step = step
#         self.dense1 = tf.keras.layers.Dense(64, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(64, activation='relu')
#         self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')
#         with tf.variable_scope('critic', reuse=tf.AUTO_REUSE):
#             self.q_value, self.q_value_next = self.get_q_value()
#         with tf.variable_scope('critic', reuse=None):
#             self.loss_dqn, self.loss_sac, self.advantage = self.compute_loss_dqn()
#         with tf.variable_scope('critic', reuse=None):
#             self.loss_dqn_, self.loss_sac_, self.advantage_ = self.compute_loss_sac()
#         self.global_step = tf.Variable(0)
#         self.learning_rate = tf.train.exponential_decay(lr, global_step=self.global_step, decay_steps=decay,
#                                                         decay_rate=lr_dc, staircase=True)
#         self.opt_dqn = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_dqn, global_step=self.global_step)
#         self.opt_sac = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_sac, global_step=self.global_step)
#         self.opt_dqn_ = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_dqn_,
#                                                                             global_step=self.global_step)
#         self.opt_sac_ = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_sac_,
#                                                                             global_step=self.global_step)
#         self.opt = tf.train.AdamOptimizer(self.learning_rate)
#         gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
#         config = tf.ConfigProto(gpu_options=gpu_options)
#         config.gpu_options.allow_growth = True
#         self.sess = tf.Session(config=config)
#         self.sess.run(tf.global_variables_initializer())
#
#     def get_q_value(self):
#         x = self.dense2(self.dense1(self.state))
#         x = self.output_layer(x)
#
#         y = self.dense2(self.dense1(self.next_state))
#         y = self.output_layer(y)
#         return x, y
#
#     def compute_loss_dqn(self):
#         # Calculate the target Q-value using the Bellman equation
#         target_q = self.reward+self.gamma * tf.reduce_max(self.q_value_next, axis=1)
#         # Q-value for the selected action
#         selected_action_q = tf.reduce_sum(tf.multiply(self.q_value, tf.one_hot(self.action, self.num_actions)), axis=1)
#         advantage = selected_action_q - target_q
#         # Calculate the mean squared loss
#         loss = tf.reduce_mean(tf.square(advantage))
#
#         loss_ = self.actor_loss+loss
#
#         return loss, loss_, advantage
#
#     def compute_loss_sac(self):
#         # target Q-value using the Bellman equation
#         target_q = self.reward+self.gamma * tf.reduce_max(self.q_value, axis=1)
#         # Q-value for the selected action
#         selected_action_q = tf.reduce_sum(tf.multiply(self.q_value_next, tf.one_hot(self.action, self.num_actions)),
#                                           axis=1)
#         advantage = target_q-selected_action_q
#         # mean squared td error loss
#         loss = tf.reduce_mean(tf.square(advantage))
#         loss_ = self.actor_loss+loss
#
#         return loss, loss_, advantage
#
#     def run(self, fetches, state, next_state, reward, action, actor_loss):
#         return self.sess.run(fetches, feed_dict={self.state: state, self.next_state: next_state, self.reward: reward,
#                                                  self.action: action,
#                                                  self.actor_loss: actor_loss})
