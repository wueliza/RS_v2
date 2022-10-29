import tensorflow.compat.v1 as tf
import gym
import numpy as np
import time
import collections

np.random.seed(int(time.time()))
COST_TO_CLOUD = 15
total_edge = 3


class MEC_network:
    def __init__(self, num_nodes, Q_SIZE, node_num):
        self.node_num = node_num
        self.num_nodes = num_nodes
        self.CRB = 5
        self.q_state = np.random.choice(5)
        self.weight_q = 1
        self.weight_d = COST_TO_CLOUD
        self.weight_s = 1
        self.Q_SIZE = Q_SIZE
        self.p_a = 0

    def reset(self):
        self.CRB = 5
        s = np.hstack((self.q_state, self.CRB))
        return s

    def step(self, share_action, price, work_type):    # not ready
        q_delay = self.q_state if self.q_state < self.Q_SIZE else self.Q_SIZE

        total_job = 0
        local_job = [0, 0]
        local_work_type = [0 for i in range(total_edge)]
        paid = 0
        for i in range(total_edge):
            local_job[work_type[i]] += share_action[i][self.node_num]
            local_work_type[i] = work_type[i]
            total_job += local_job[work_type[i]]*(local_work_type[i]+1)

        self.q_state = total_job - self.CRB
        self.q_state = self.q_state if self.q_state > 0 else 0
        self.q_state = self.q_state if self.q_state < self.Q_SIZE else self.Q_SIZE

        d_delay = self.q_state - self.Q_SIZE if self.q_state > self.Q_SIZE else 0

        reward = float(q_delay + self.weight_d * d_delay)

        self.CRB = 5

        s_ = np.hstack((self.CRB, self.q_state))
        total_work_ = self.q_state

        avg_delay = (1 / (self.Q_SIZE - self.CRB)) if self.Q_SIZE - self.q_state != 0 else 15

        return s_, total_work_, reward, d_delay, q_delay, avg_delay

