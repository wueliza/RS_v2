import tensorflow.compat.v1 as tf
import gym
import numpy as np
import time
import collections

np.random.seed(int(time.time()))
COST_TO_CLOUD = 15
total_edge = 3
class MEC_network:
    def __init__(self, task_arrival_rate, num_nodes, Q_SIZE, node_num):
        self.node_num = node_num
        self.num_nodes = num_nodes
        self.p_state = np.random.choice([4, 4])
        self.q_state = np.random.choice(5)
        self.task_arrival_rate = task_arrival_rate
        self.weight_q = 1
        self.weight_d = COST_TO_CLOUD
        self.weight_s = 1
        self.Q_SIZE = Q_SIZE
        self.p_a = 0

    def reset(self):
        self.p_state = np.random.choice([4, 4])
        s = np.ones(total_edge)
        s[self.node_num] = self.p_state
        total_work = self.q_state
        return s, total_work

    def step(self, share_action, price):
        task_arrival_rate = self.task_arrival_rate
        new_task = np.random.poisson(task_arrival_rate)

        q_delay = self.q_state if self.q_state < self.Q_SIZE else self.Q_SIZE

        local_job = 0
        paid = 0
        for i, j in range(total_edge), range(total_edge+1):
            if j == self.node_num:
                local_job += share_action[i][j]

        self.q_state = local_job - self.p_state + new_task
        self.q_state = self.q_state if self.q_state > 0 else 0
        self.q_state = self.q_state if self.q_state < self.Q_SIZE else self.Q_SIZE

        d_delay = self.q_state - self.Q_SIZE if self.q_state > self.Q_SIZE else 0

        reward = float(q_delay + self.weight_d * d_delay)

        self.p_state = np.random.choice([4, 4])

        s_ = np.hstack((self.p_state, self.q_state))
        total_work_ = self.q_state

        avg_delay = (1 / (self.Q_SIZE - self.q_state)) if self.Q_SIZE - self.q_state != 0 else 15

        return s_, total_work_, reward, d_delay, q_delay, new_task, avg_delay
