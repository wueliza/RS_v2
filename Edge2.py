# use the formula from the paper "Game 2016 Fog computing"
import math
import numpy as np
import pandas as pd
import gym
from pulp import*

total_edge = 3
COST_TO_CLOUD = 15
bins = list(np.arange(0-1/COST_TO_CLOUD, 1, 1/COST_TO_CLOUD))
bins[len(bins)-1] = 1


class Predictor(object):
    def __init__(self, n_nodes,  q_size):
        self.n_nodes = n_nodes
        self.q_size = q_size
        self.CRB = 10

    def choose_action(self, share_actions):
        price = []

        for name, task in share_actions.items():
            if name == 'cloud':
                break
            p = math.pow(task / (1.0001 * self.CRB - task), 2)
            p *= 5
            price.append(p)

        # price = pd.cut(price, bins, labels=False)
        return price


class Actor(object):
    def __init__(self, n_nodes, q_size):
        self.n_nodes = n_nodes
        self.q_size = q_size
        self.CRB = 10

    def choose_action(self, user_buy, tr):
        price = math.pow(tr/(self.CRB*user_buy - tr), 2)

        return price

    def reset(self):
        tf.reset_default_graph()


class Edge(object):
    def __init__(self, q_size, node_num, task_arrival_rate):
        self.n_nueron_ac = 5
        self.node_num = node_num
        self.q_size = q_size
        self.CRB = 10
        self.task_arrival_rate = task_arrival_rate
        self.local_actor = Actor(node_num, self.q_size)
        # self.local_critic = Critic(total_edge)
        self.local_predictor = Predictor(node_num, self.q_size)

    def distribute_work(self, price, work, p_user):     # (predict price [], form user {type: amount}, price for user)
        new_task_type = list(work.keys())[0]
        new_task = work[new_task_type]

        q_delay = price[self.node_num]
        price[self.node_num] = 0
        price = np.append(price, COST_TO_CLOUD)    # add cloud price

        trans_utility = 0
        model1 = pulp.LpProblem("value min", sense=LpMinimize)
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

    def price_user(self, u0_buy, tr):
        price = math.pow(tr/(self.CRB*u0_buy - tr), 2)

        return price
