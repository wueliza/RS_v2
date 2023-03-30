# user without NN
import math

import tensorflow.compat.v1 as tf
import gym
import numpy as np
import pandas as pd
import random
from pulp import*

GAMMA = 0.9
COST_TO_CLOUD = 15


class User(object):  # contain a local actor, critic, global critic
    def __init__(self, task_arrival_rate, edge_num, q_size=0, sess=None):
        self.sess = sess
        self.edge_num = edge_num
        self.task_arrival_rate = task_arrival_rate
        self.CRB = 2  # computing resource block
        self.q_state = 0  # queueing state
        self.q_size = q_size
        self.work = [0, 0]  # [0] = task1 [1] = task2
        self.work_type = np.random.choice(2)


    def need_CRB(self, edge_price):
        task_arrival_rate = self.task_arrival_rate
        new_task = np.random.poisson(task_arrival_rate)
        work_type = self.work_type

        buy = task_arrival_rate*( 1/self.CRB + 1/(self.CRB - math.sqrt(edge_price)))
        trans = min(new_task*(work_type+1) - self.CRB, buy)
        trans = trans / (work_type+1)

        return {work_type:trans}



    def step(self, trans_task, edge_price):



    # def step(self, edge_price):
    #     task_arrival_rate = self.task_arrival_rate
    #     new_task = np.random.poisson(task_arrival_rate)
    #
    #     # user does it self
    #     overflow = new_task * (self.work_type + 1) - self.CRB
    #     overflow = overflow if overflow > 0 else 0
    #     do_self_utility = overflow * COST_TO_CLOUD
    #
    #     # transfer all work to edge
    #     trans_utility = new_task * (self.work_type + 1) * edge_price
    #
    #     # allocate work
    #     model1 = pulp.LpProblem("value max", sense=LpMinimize)
    #     t0 = pulp.LpVariable('t0', lowBound=0, cat=LpInteger)
    #     t1 = pulp.LpVariable('t1', lowBound=0, cat=LpInteger)
    #     tcloud = pulp.LpVariable('tcloud', lowBound=0, cat=LpInteger)
    #
    #     model1 += ((t0 * (self.work_type + 1)) - self.CRB)*COST_TO_CLOUD + t1 * (self.work_type + 1) * edge_price + tcloud * (self.work_type + 1) * COST_TO_CLOUD
    #     model1 += (t0 * (self.work_type + 1) - self.CRB)*COST_TO_CLOUD + t1 * (self.work_type + 1) * edge_price + tcloud * (self.work_type + 1) * COST_TO_CLOUD >= 0
    #     model1 += t0 + t1 + tcloud == new_task * (self.work_type + 1)
    #     model1.solve(PULP_CBC_CMD(msg=0))
    #     allo_utility = pulp.value(model1.objective)
    #
    #     # print(model1)
    #     # for v in model1.variables():
    #     #     print(f'{v.name} = {v.value()}')
    #     # print(allo_utility)
    #
    #     trans_work = 0
    #     work = 0
    #     utility = min(do_self_utility, trans_utility, allo_utility)
    #     if utility == do_self_utility:
    #         trans_work = 0
    #         overflow = new_task * (self.work_type + 1) - self.CRB
    #         overflow = overflow if overflow > 0 else 0
    #         work = new_task
    #     elif utility == trans_utility:
    #         trans_work = new_task
    #         overflow, work = 0, 0
    #     elif utility == allo_utility:
    #         trans_work = value(t1)
    #         work = new_task - trans_work / (self.work_type + 1)
    #         overflow = work * (self.work_type + 1) - self.CRB
    #
    #     return utility, overflow, new_task, {self.work_type: trans_work}, work
