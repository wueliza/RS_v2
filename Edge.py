import tensorflow.compat.v1 as tf
import gym
import numpy as np
from pulp import*
import pandas as pd

GAMMA = 0.9
total_edge = 3
COST_TO_CLOUD = 15
bins = list(np.arange(0, 1, 1/COST_TO_CLOUD))
bins[len(bins)-1] = 1


class Predictor(object):
    def __init__(self, scope, sess, n_node, lr=0.001):
        self.sess = sess
        self.lr = lr
        n_features = n_node + 1
        self.state = tf.placeholder(tf.float32, [1, total_edge+1], "Action")
        self.value_ = tf.placeholder(tf.float32, [1, total_edge], "NextValue")
        self.reward = tf.placeholder(tf.float32, None, "pre_reward")
        self.t = 1

        with tf.variable_scope(scope + 'Predictor'):
            l1 = tf.layers.dense(
                inputs=self.state,
                units=10,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.value = tf.layers.dense(
                inputs=l1,
                units=total_edge,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='Value'
            )

        with tf.variable_scope(scope + 'squared_TD_error'):
            self.td_error = self.value - self.reward  # reality - predict
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval

        with tf.variable_scope(scope + 'train'):
            self.train_op = tf.train.RMSPropOptimizer(lr).minimize(self.loss)  # for under 10 nodes .01

    def choose_action(self, state):
        state = np.reshape(state, (1, total_edge+1))
        value = self.sess.run(self.value, {self.state: state})

        price = pd.cut(value.flatten(), bins, labels=False)
        return price

    def learn(self, s, r, s_):
        s, s_ = np.reshape(s, (1, total_edge+1)), np.reshape(s_, (1, total_edge+1))
        v_ = self.sess.run(self.value, {self.state: s_})
        td_error, loss, _ = self.sess.run([self.td_error, self.loss, self.train_op],
                                          {self.state: s, self.value_: v_, self.reward: r})
        self.t += 1
        return td_error, loss

    def reset(self):
        tf.reset_default_graph()


class Actor(object):
    def __init__(self, scope, sess, n_nodes, lr=0.001, q_size=10):
        self.sess = sess
        self.lr = lr
        self.t = 1
        self.n_nodes = n_nodes
        self.n_actions = 3
        self.n_features = 3  # pStates, qStates, and cStates
        self.state = tf.placeholder(tf.float32, [1, total_edge+2], "state")  # Try different dimensions
        self.epsilon = 0.9
        self.action = tf.placeholder(tf.float32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.q_size = q_size

        with tf.variable_scope(scope + 'Actor'):
            l1 = tf.layers.dense(
                inputs=self.state,
                units=10,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(  # output layer
                inputs=l1,
                units=1,  # output units
                activation=tf.nn.sigmoid,  ###############
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
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.exp_v * .5)  # -.2  # minimize(-exp_v) = maximize(exp_v) #10.5
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

    def choose_action(self, total_task):
        value = self.sess.run(self.acts_prob, feed_dict={self.state: total_task[np.newaxis, :]})
        price = pd.cut(value.flatten(), bins, labels=False)
        return price

    def reset(self):
        tf.reset_default_graph()


class Critic(object):
    def __init__(self, scope, sess, n_nodes, lr=0.001):
        self.sess = sess
        self.lr = lr
        self.t = 1
        n_features = 3
        self.s = tf.placeholder(tf.float32, [1, 1], "state")
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
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope(scope + 'squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope(scope + 'train'):
            self.train_op = tf.train.RMSPropOptimizer(lr).minimize(self.loss * .5)  # for under 10 nodes .01

            # self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.loss*.05) #.5
            # self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v * 10.5)  # minimize(-exp_v) = maximize(exp_v)
            # self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(self.loss*0.001)
            # self.train_op = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = np.reshape(s, (1, 1)), np.reshape(s_, (1, 1))
        v = self.sess.run(self.v, {self.s: s})
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, loss, _ = self.sess.run([self.td_error, self.loss, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        # self.lr = min(1, self.lr * math.pow(1.0000005, self.t))
        self.t += 1
        return td_error, v_, loss, v

    def reset(self):
        tf.reset_default_graph()


class Edge(object):  # contain a local actor, critic, global critic
    def __init__(self, scope, lar=0.001, lcr=0.001, q_size=50, sess=None, node_num=0):
        self.n_nueron_ac = 5
        self.node_num = node_num
        self.sess = sess
        self.la_r = lar
        self.lc_r = lcr
        # self.la_s = tf.placeholder(tf.float32, 1, [None, N_S], 'la_s')
        self.epsilon = 0.8
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.q_size = q_size
        self.local_actor = Actor(scope, self.sess, 2, self.la_r, self.q_size)
        self.local_critic = Critic(scope, self.sess, 2, self.lc_r)
        self.local_predictor = Predictor(scope, self.sess, 2, self.lc_r)

    def distribute_work(self, price, work, p_user):     # (predict price [], form user {type: amount}, price for user)
        new_task_type = list(work.keys())[0]
        new_task = work[new_task_type]

        q_delay = price[self.node_num]
        price[self.node_num] = 0
        price = np.append(price, COST_TO_CLOUD)    # add cloud price

        trans_utility = 0
        model1 = pulp.LpProblem("value min", sense=LpMaximize)
        t0 = pulp.LpVariable('t0', lowBound=0, cat='Integer')
        t1 = pulp.LpVariable('t1', lowBound=0, cat='Integer')
        t2 = pulp.LpVariable('t2', lowBound=0, cat='Integer')
        tcloud = pulp.LpVariable('tcloud', lowBound=0, cat='Integer')

        model1 += t0 * price[0] + t1 * price[1] + t2 * price[2] + tcloud * price[3]
        model1 += t0 * price[0] + t1 * price[1] + t2 * price[2] + tcloud * price[3] >= 0
        model1 += t0+t1+t2+tcloud == new_task
        model1.solve(PULP_CBC_CMD(msg=0))

        shared_r = {}
        shared = {}
        for v, j in zip(model1.variables(), range(total_edge+1)):
            if j == self.node_num:
                shared_r['self'] = v.varValue
            elif j == total_edge:
                shared_r['cloud'] = v.varValue
            else:
                shared_r[f'edge{j}'] = v.varValue
                shared[f'edge{j}'] = {new_task_type: v.varValue}
            trans_utility += v.varValue * price[j]

        price_ = p_user if trans_utility < q_delay else p_user * 1.5
        paid = pulp.value(model1.objective)
        return shared_r, price_, shared, paid

