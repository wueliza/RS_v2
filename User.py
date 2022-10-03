import tensorflow.compat.v1 as tf
import gym
import numpy as np



class User(object):  # contain a local actor, critic, global critic
    def __init__(self, scope, lar=0.001, lcr=0.001, q_size=10, sess=None):
        self.n_nueron_ac = 5
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
        # self.local_predictor = Predictor(scope, self.sess, 2, self.lc_r)

