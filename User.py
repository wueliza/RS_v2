import tensorflow.compat.v1 as tf
import gym
import numpy as np
GAMMA = 0.9


class Actor(object):
    def __init__(self, scope, sess, edge_num, lr=0.001, q_size=10):
        self.sess = sess
        self.lr = lr
        self.t = 1
        self.edge_num = edge_num  # under which edge
        self.n_actions = 2
        self.n_features = 3  # pStates, qStates, and cStates
        self.state = tf.placeholder(tf.float32, [1, 3], "state")  # the price of edge
        self.epsilon = 0.9
        self.action = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.q_size = q_size

        with tf.variable_scope(scope + 'Actor'):
            l1 = tf.layers.dense(
                inputs=self.state,
                units=10,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(  # output layer
                inputs=l1,
                units=1,  # output units
                activation=tf.nn.sigmoid,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )
        with tf.variable_scope(scope + 'exp_v'):
            # log_prob = tf.log(self.acts_prob[0, self.a])
            log_prob = tf.log(self.acts_prob)  # 自然對數函數
            self.exp_v = tf.reduce_mean(
                tf.math.reduce_sum(tf.math.multiply(log_prob, self.td_error)))  # advantage (TD_error) guided loss

        with tf.variable_scope(scope + 'train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(
                -self.exp_v * .5)  # -.2  # minimize(-exp_v) = maximize(exp_v) #10.5
            # Adam optimization algorithm (for stochastic optimization)
            # self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(-self.exp_v*0.00005)
            # self.train_op = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(-self.exp_v*0.005)
            # self.train_op = tf.train.RMSPropOptimizer(lr).minimize(-self.exp_v*.05)

    def learn(self, s, a, td):
        state = s[np.newaxis, :]
        feed_dict = {self.state: state, self.action: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        self.t += 1

        return exp_v

    def choose_action(self, s):
        price = self.sess.run(self.acts_prob, feed_dict={self.state: s[np.newaxis, :]})

        return price

    def reset(self):
        tf.reset_default_graph()


class Critic(object):
    def __init__(self, scope, sess, edge_num, lr=0.001):
        self.sess = sess
        self.lr = lr
        self.t = 1
        self.edge_num = edge_num
        n_features = 3
        self.s = tf.placeholder(tf.float32, [1, 3], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope(scope + 'Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units #50
                # activation=tf.nn.relu,  # None
                activation=tf.nn.tanh,
                # tf.nn.tanh
                # tf.nn.selu
                # tf.nn.softplus
                # tf.nn.elu
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope(scope + 'squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope(scope + 'train'):
            self.train_op = tf.train.RMSPropOptimizer(lr).minimize(-self.loss * .5)  # for under 10 nodes .01

            # self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.loss*.05) #.5
            # self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v * 10.5)  # minimize(-exp_v) = maximize(exp_v)
            # self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(self.loss*0.001)
            # self.train_op = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v = self.sess.run(self.v, {self.s: s})
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, loss, _ = self.sess.run([self.td_error, self.loss, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        # self.lr = min(1, self.lr * math.pow(1.0000005, self.t))
        self.t += 1
        return td_error, v_, loss, v

    def reset(self):
        tf.reset_default_graph()


class User(object):  # contain a local actor, critic, global critic
    def __init__(self, scope, task_arrival_rate, edge_num, lar=0.001, lcr=0.001, q_size=10, sess=None):
        self.n_nueron_ac = 5
        self.sess = sess
        self.edge_num = edge_num
        self.task_arrival_rate = task_arrival_rate
        self.la_r = lar
        self.lc_r = lcr
        # self.la_s = tf.placeholder(tf.float32, 1, [None, N_S], 'la_s')
        self.epsilon = 0.8
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.CRB = np.random.choice(5)  # computing state
        self.q_state = np.random.choice(5)  # queueing state
        self.q_size = q_size
        self.local_actor = Actor(scope, self.sess, self.edge_num, self.la_r, self.q_size)
        self.local_critic = Critic(scope, self.sess, self.edge_num, self.lc_r)
        self.work = [0, 0]  # [0] = task1 [1] = task2
    def step(self, action, edge_price):
        task_arrival_rate = self.task_arrival_rate
        new_task1 = np.random.poisson(task_arrival_rate)
        new_task2 = np.random.poisson(task_arrival_rate)

        # random choose to pass task1 or task2 to edge
        self.work[0] += new_task1
        self.work[1] += new_task2

        transit_task = 0 if np.random.rand() > 0.5 else 1
        tw = round(self.work[transit_task]*action)
        self.work[transit_task] -= tw
        transit_work = {transit_task: tw}

        # local
        self.CRB = 2
        local_total_work = 0
        for i in range(len(self.work)):
            local_total_work += self.work[i]*(i+1)
        self.q_state = local_total_work - self.CRB
        self.q_state = self.q_state if self.q_state > 0 else 0
        self.q_state = self.q_state if self.q_state < self.q_size else self.q_size
        for i in range(self.CRB):
            if self.work[0] > 0:
                self.work[0] -= 1
            elif self.work[1] > 0:
                self.work[1] -= 1
            else:
                break

        local_overflow = local_total_work - self.q_size if local_total_work > self.CRB else 0
        q_delay = self.q_state - self.CRB if self.q_state > self.CRB else 0
        d_delay = local_overflow + q_delay

        utility = transit_work[transit_task] * edge_price + d_delay
        s_ = np.hstack((self.q_state, self.CRB, d_delay))

        return transit_work, utility, s_, task_arrival_rate, self.work, local_overflow, q_delay
