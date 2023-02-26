import tensorflow.compat.v1 as tf
import gym
import numpy as np
import time
import collections
import pandas as pd

np.random.seed(int(time.time()))
COST_TO_CLOUD = 15
total_edge = 3
bins = list(np.arange(0, 1, 1/COST_TO_CLOUD))
bins[len(bins)-1] = 1


class MEC_network:
    def __init__(self, num_nodes, Q_SIZE, node_num):
        self.node_num = node_num
        self.num_nodes = num_nodes
        self.CRB = 10
        self.q_state = np.random.choice(5)
        self.weight_q = 1
        self.weight_d = COST_TO_CLOUD
        self.weight_s = 1
        self.Q_SIZE = Q_SIZE
        self.p_a = 0

    def reset(self):
        self.CRB = 10
        s = np.hstack((self.q_state, self.CRB))
        return s

    def step(self, share_action, work, price):

        q_delay = self.q_state if self.q_state < self.Q_SIZE else self.Q_SIZE

        total_job = 0
        income = 0
        local_overflow = 0
        for n, v in work.items():
            income += v * price[f'edge{self.node_num}']
            if self.Q_SIZE - total_job - self.CRB < v * (n+1):
                local_overflow += v
            else:
                total_job += v * (n+1)

        paid = 0
        for n, v in share_action[f'edge{self.node_num}'].items():
            if n != 'self' and n != 'cloud':
                paid += v * price[n]
            elif n == 'cloud':
                paid += v * COST_TO_CLOUD

        self.q_state = total_job - self.CRB
        self.q_state = self.q_state if self.q_state > 0 else 0
        self.q_state = self.q_state if self.q_state < self.Q_SIZE else self.Q_SIZE

        local_overflow = total_job - self.Q_SIZE if total_job > self.Q_SIZE else 0
        d_delay = local_overflow * COST_TO_CLOUD + q_delay
        utility = d_delay + paid - income

        s_ = np.hstack((self.q_state, self.CRB))
        total_work_ = self.q_state

        avg_delay = (1 / (self.Q_SIZE - self.CRB)) if self.Q_SIZE - self.q_state != 0 else 15

        return s_, total_work_, utility, d_delay, q_delay, avg_delay, paid, total_job, local_overflow, income

