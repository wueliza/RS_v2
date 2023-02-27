import os
import random
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
from Work import MEC_network
from Edge import Edge
from User2 import User
import gym
import time
import random
import math
import collections
import matplotlib.pyplot as plt
from collections import Counter

tf.disable_eager_execution()  # 禁用默認的即時執行模式(tensorflow 1轉2 需要)
GAMMA = 0.9
ALPHA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.01  # learning rate for actor
LR_C = 0.05  # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
COST_TO_CLOUD = 15

# State and Action Space

n_mobile_user = 3
N_mec_edge = 2
N_A = N_mec_edge + 1

N_S = 5  # Latency State or Latency State + action phi
output = 'output.txt'
f = open(output, 'w')

def run(tr):
    import time

    # State and Action Space

    N_mec_edge = 3
    N_A = N_mec_edge + 1

    total_delay = []
    total_jobs = []
    total_drop = []
    total_time = 100
    q_len0 = 0
    q_len1 = 0
    q_len2 = 0
    total_r0 = 0
    total_r1 = 0
    total_r2 = 0

    user_q_len0 = 0
    user_q_len1 = 0
    user_q_len2 = 0
    user_total_r0 = 0
    user_total_r1 = 0
    user_total_r2 = 0

    total_q_delay = 0
    total_drop = 0
    total_utility = 0
    total_state_value = 0
    total_s_delay = 0

    SESS = tf.Session()
    # Three edge networks plus a cloud
    # create networks
    mec_0 = MEC_network(num_nodes=1, Q_SIZE=50, node_num=0)
    s_0 = mec_0.reset()  # s = [q_state, CRB]
    edge_0 = Edge(scope='e' + str(0), lar=0.001, lcr=0.01, q_size=50, sess=SESS, node_num=0)

    mec_1 = MEC_network(num_nodes=2, Q_SIZE=50, node_num=1)
    s_1 = mec_1.reset()
    edge_1 = Edge(scope='e' + str(1), lar=0.001, lcr=0.01, q_size=50, sess=SESS, node_num=1)

    mec_2 = MEC_network(num_nodes=3, Q_SIZE=50, node_num=2)
    s_2 = mec_2.reset()
    edge_2 = Edge(scope='e' + str(2), lar=0.001, lcr=0.01, q_size=50, sess=SESS, node_num=2)

    # user_0 = User(scope='u' + str(0), task_arrival_rate=tr, edge_num=0, lar=0.001, lcr=0.01, q_size=0, sess=SESS)
    # user_s_0 = np.hstack((user_0.q_state, user_0.CRB))  # s = [q_state, CRB]
    # user_1 = User(scope='u' + str(1), task_arrival_rate=tr, edge_num=1, lar=0.001, lcr=0.01, q_size=0, sess=SESS)
    # user_s_1 = np.hstack((user_1.q_state, user_1.CRB))  # s = [q_state, CRB]
    # user_2 = User(scope='u' + str(2), task_arrival_rate=tr, edge_num=2, lar=0.001, lcr=0.01, q_size=0, sess=SESS)
    # user_s_2 = np.hstack((user_2.q_state, user_2.CRB))  # s = [q_state, CRB]

    user_0 = User(task_arrival_rate=tr,edge_num=0, q_size=0, sess=SESS)
    user_s_0 = np.hstack((user_0.q_state, user_0.CRB))  # s = [q_state, CRB]
    user_1 = User(task_arrival_rate=tr, edge_num=1, q_size=0, sess=SESS)
    user_s_1 = np.hstack((user_1.q_state, user_1.CRB))  # s = [q_state, CRB]
    user_2 = User(task_arrival_rate=tr, edge_num=2, q_size=0, sess=SESS)
    user_s_2 = np.hstack((user_2.q_state, user_2.CRB))  # s = [q_state, CRB]
    SESS.run(tf.global_variables_initializer())

    # store the distribution of the task to other edge
    # shared_ations = np.zeros((N_mec_edge, N_mec_edge+1, 2))

    shared_action = {}
    for k in range(N_mec_edge):
        shared_action[f'edge{k}'] = {}
        shared_action[f'edge{k}']['self'] = 0
        for j in range(N_mec_edge):
            if j == k:
                continue
            else:
                shared_action[f'edge{k}'][f'edge{j}'] = 0
        shared_action[f'edge{k}']['cloud'] = 0
    # local_work_0 = [0, 0]
    # local_work_1 = [0, 0]
    # local_work_2 = [0, 0]
    work = {}
    for k in range(N_mec_edge):
        work[f'edge{k}'] = {0: 1, 1: 0}

    ap0 = []
    ap1 = []
    ap2 = []

    # q_len0 += s_0[0]
    # q_len1 += s_1[0]
    # q_len2 += s_2[0]
    # print(f'\nq0 = {q_len0}  q1 = {q_len1}  q2 = {q_len2}', file=f)

    latency0 = []
    latency1 = []
    latency2 = []
    user_la_0 = []
    user_la_1 = []
    user_la_2 = []
    pp0 = []
    pp1 = []
    pp2 = []
    for i in range(total_time):
        print("time = ", i)
        print("\ntime = ", i, file=f)
        # q_len0 += s_0[0]
        # q_len1 += s_1[0]
        # q_len2 += s_2[0]
        # print(f'\nq0 = {q_len0}  q1 = {q_len1}  q2 = {q_len2}', file=f)

        # predict the other edge's price bace on the work distribution last time
        PD_other_price_0 = edge_0.local_predictor.choose_action(shared_action['edge0']).flatten()
        PD_other_price_1 = edge_1.local_predictor.choose_action(shared_action['edge1']).flatten()
        PD_other_price_2 = edge_2.local_predictor.choose_action(shared_action['edge2']).flatten()
        print(f'PD_other_price_0 = {PD_other_price_0}  PD_other_price_1 = {PD_other_price_1}  PD_other_price_2 = {PD_other_price_2}', file=f)

        # merge the queue state and cpu
        s_0 = np.hstack((s_0, PD_other_price_0))
        s_1 = np.hstack((s_1, PD_other_price_1))
        s_2 = np.hstack((s_2, PD_other_price_2))
        print(f's0 = {s_0} s1 = {s_1} s2 = {s_2}', file=f)

        # actor determine the price for user of this state
        p0 = edge_0.local_actor.choose_action(s_0).flatten()[0]
        p1 = edge_1.local_actor.choose_action(s_1).flatten()[0]
        p2 = edge_2.local_actor.choose_action(s_2).flatten()[0]
        print(f'p0 = {p0}  p1 = {p1}  p2 = {p2}', file=f)

        # user
        # user_s_0 = np.hstack((user_s_0, p0))
        # user_s_1 = np.hstack((user_s_1, p1))
        # user_s_2 = np.hstack((user_s_2, p2))
        # print(f'\nuser_s0 = {user_s_0}  user_s1 = {user_s_1}  user_s2 = {user_s_2}', file=f)
        #
        # user_0_action = user_0.local_actor.choose_action(user_s_0).flatten()[0]
        # user_1_action = user_1.local_actor.choose_action(user_s_1).flatten()[0]
        # user_2_action = user_2.local_actor.choose_action(user_s_2).flatten()[0]
        # print(f'user0 action = {user_0_action}  user1 action = {user_1_action}  user2 action = {user_2_action}', file=f)

        # user_0_task, user_0_utility, user_s_0_, u0_tr, u0_work, u0_overflow, u0_q_delay, u0_nt, u0_nt1, u0_nt2 = user_0.step(user_0_action, p0)
        # user_1_task, user_1_utility, user_s_1_, u1_tr, u1_work, u1_overflow, u1_q_delay, u1_nt, u1_nt1, u1_nt2 = user_1.step(user_1_action, p1)
        # user_2_task, user_2_utility, user_s_2_, u2_tr, u2_work, u2_overflow, u2_q_delay, u2_nt, u2_nt1, u2_nt2 = user_2.step(user_2_action, p2)

        user_0_task, user_0_utility, user_s_0_, u0_tr, u0_work, u0_overflow, u0_q_delay, u0_nt, u0_nt1, u0_nt2 = user_0.step(p0)
        user_1_task, user_1_utility, user_s_1_, u1_tr, u1_work, u1_overflow, u1_q_delay, u1_nt, u1_nt1, u1_nt2 = user_1.step(p1)
        user_2_task, user_2_utility, user_s_2_, u2_tr, u2_work, u2_overflow, u2_q_delay, u2_nt, u2_nt1, u2_nt2 = user_2.step(p2)
        print(f'user0 trans task = {user_0_task}  utility = {user_0_utility}  user_s_ = {user_s_0_}  tr = {u0_tr}  work = {u0_work}  overflow = {u0_overflow}  q_delay = {u0_q_delay}  new task  = {u0_nt}  new task 1 = {u0_nt1} new task 2 = {u0_nt2}', file=f)
        print(f'user1 trans task = {user_1_task}  utility = {user_1_utility}  user_s_ = {user_s_1_}  tr = {u1_tr}  work = {u1_work}  overflow = {u1_overflow}  q_delay = {u1_q_delay}  new task  = {u1_nt}  new task 1 = {u1_nt1} new task 2 = {u1_nt2}', file=f)
        print(f'user2 trans task = {user_2_task}  utility = {user_2_utility}  user_s_ = {user_s_2_}  tr = {u2_tr}  work = {u2_work}  overflow = {u2_overflow}  q_delay = {u2_q_delay}  new task  = {u2_nt}  new task 1 = {u2_nt1} new task 2 = {u2_nt2}', file=f)

        user_total_r0 += u0_q_delay
        user_total_r1 += u1_q_delay
        user_total_r2 += u2_q_delay
        user_q_len0 += user_0.q_state
        user_q_len1 += user_1.q_state
        user_q_len2 += user_2.q_state

        # user_td_error_0 = user_0.local_critic.learn(user_s_0, user_0_utility, user_s_0_)
        # user_td_error_1 = user_1.local_critic.learn(user_s_1, user_1_utility, user_s_1_)
        # user_td_error_2 = user_2.local_critic.learn(user_s_2, user_2_utility, user_s_2_)
        #
        # user_0.local_actor.learn(user_s_0, user_0_action, user_td_error_0)
        # user_1.local_actor.learn(user_s_1, user_1_action, user_td_error_1)
        # user_2.local_actor.learn(user_s_2, user_2_action, user_td_error_2)

        # user_s_0 = user_s_0_[:len(user_s_0_)-1]
        # user_s_1 = user_s_1_[:len(user_s_1_)-1]
        # user_s_2 = user_s_2_[:len(user_s_2_)-1]

        # user pass the work to edge user task {type: amount}
        # local_work_0[list(user_0_task.items())[0][0]] += list(user_0_task.items())[0][1]
        # local_work_1[list(user_1_task.items())[0][0]] += list(user_1_task.items())[0][1]
        # local_work_2[list(user_2_task.items())[0][0]] += list(user_2_task.items())[0][1]
        # print(f'\nlocal work 0 = {local_work_0}  local work 1 = {local_work_1}  local work 2 = {local_work_2}', file=f)

        # distribute the work base on the predict price of the other edge
        PD_other_price_0[0] = mec_0.q_state if mec_0.q_state < mec_0.Q_SIZE else mec_0.Q_SIZE
        PD_other_price_1[1] = mec_1.q_state if mec_1.q_state < mec_1.Q_SIZE else mec_1.Q_SIZE
        PD_other_price_2[2] = mec_2.q_state if mec_2.q_state < mec_2.Q_SIZE else mec_2.Q_SIZE
        shared_action['edge0'], actual_p0, shared_0, paid_0 = edge_0.distribute_work(PD_other_price_0, user_0_task, p0)
        shared_action['edge1'], actual_p1, shared_1, paid_1 = edge_1.distribute_work(PD_other_price_1, user_1_task, p1)
        shared_action['edge2'], actual_p2, shared_2, paid_2 = edge_2.distribute_work(PD_other_price_2, user_2_task, p2)
        print(f'actual price p0 = {actual_p0}  p1 = {actual_p1}  p2 = {actual_p2} \nshared action: ', file=f)
        print(shared_action, file=f)
        print(f'transfer task: \nedge0: {shared_0}\nedge1: {shared_1} \nedge2: {shared_2}', file=f)

        # transfer task to other edges
        work['edge0'].update(Counter(shared_1['edge0']) + Counter(shared_2['edge0']))
        work['edge1'].update(Counter(shared_0['edge1']) + Counter(shared_2['edge1']))
        work['edge2'].update(Counter(shared_1['edge2']) + Counter(shared_0['edge2']))
        print(f'work:\n{work}', file=f)

        # collect the actual price of all edge
        price = {'edge0': actual_p0, 'edge1': actual_p1, 'edge2': actual_p2}  # actual price
        ap0.append(actual_p0)
        ap1.append(actual_p1)
        ap2.append(actual_p2)

        # calculate real utility
        s_0_, total_work_0_, r_0, d_0, q_d_0, avg_delay_0, paid_0, tj0, overflow0, income_0 = mec_0.step(shared_action['edge0'], work['edge0'], price)  # s_, total_work_, reward, d_delay, q_delay, new_task, avg_delay
        s_1_, total_work_1_, r_1, d_1, q_d_1, avg_delay_1, paid_1, tj1, overflow1, income_1 = mec_1.step(shared_action['edge1'], work['edge1'], price)
        s_2_, total_work_2_, r_2, d_2, q_d_2, avg_delay_2, paid_2, tj2, overflow2, income_2 = mec_2.step(shared_action['edge2'], work['edge2'], price)

        q_len0 += tj0
        q_len1 += tj1
        q_len2 += tj2
        print(f'q0 = {q_len0}  q1 = {q_len1}  q2 = {q_len2}', file=f)
        print(f'edge0: s_ = {s_0_} work_ = {total_work_0_} r = {r_0} d = {d_0} qd = {q_d_0} ad = {avg_delay_0} paid = {paid_0} total_job = {tj0} overflow = {overflow0} income = {income_0}', file=f)
        print(f'edge1: s_ = {s_1_} work_ = {total_work_1_} r = {r_1} d = {d_1} qd = {q_d_1} ad = {avg_delay_1} paid = {paid_1} total_job = {tj1} overflow = {overflow1} income = {income_1}', file=f)
        print(f'edge2: s_ = {s_2_} work_ = {total_work_2_} r = {r_2} d = {d_2} qd = {q_d_2} ad = {avg_delay_2} paid = {paid_2} total_job = {tj2} overflow = {overflow2} income = {income_2}', file=f)

        # if r_0 < 0 or r_1 < 0 or r_2 < 0:
        #     print(r_0, r_1)
        #     print("stop1")
        #     exit()

        # edge actual reward
        # r_0 = r_0 - user_0_utility
        # r_1 = r_1 - user_1_utility
        # r_2 = r_2 - user_2_utility
        r_0 = r_0
        r_1 = r_1
        r_2 = r_2
        print(f'r0 = {r_0}  r1 = {r_1}  r2 = {r_2}', file=f)
        # if r_0 < 0 or r_1 < 0 or r_2 < 0:
        #     print("stop2")
        #     exit()

        # total_edge_q_len[1][0] += local_work_0[1] + local_work_1[1] + local_work_2[1]
        # total_edge_q_len[2][0] += local_work_0[2] + local_work_1[2] + local_work_2[2]

        td_error_0, v_0, _, v_0_ = edge_0.local_critic.learn(p0, r_0, actual_p0)  # td_error = the actual price - the predict price
        td_error_1, v_1, _, v_1_ = edge_1.local_critic.learn(p1, r_1, actual_p1)
        td_error_2, v_2, _, v_2_ = edge_1.local_critic.learn(p2, r_2, actual_p2)

        edge_0.local_actor.learn(s_0, p0, td_error_0)
        edge_1.local_actor.learn(s_1, p1, td_error_1)
        edge_2.local_actor.learn(s_2, p2, td_error_2)

        PD_other_price_0 = np.append(PD_other_price_0, COST_TO_CLOUD)
        PD_other_price_1 = np.append(PD_other_price_1, COST_TO_CLOUD)
        PD_other_price_2 = np.append(PD_other_price_2, COST_TO_CLOUD)
        edge_0.local_predictor.learn(PD_other_price_0, r_0, price)  # actual price & predict price
        edge_1.local_predictor.learn(PD_other_price_1, r_1, price)  # price -> dict PD -> list 要改
        edge_2.local_predictor.learn(PD_other_price_2, r_2, price)

        s_0 = s_0_
        s_1 = s_1_
        s_2 = s_2_

        ###########################
        edge_0.local_actor.lr = min(1, edge_0.local_actor.lr * math.pow(1.000001, i))  # learning rate
        edge_0.local_critic.lr = min(1, edge_0.local_critic.lr * math.pow(1.000001, i))

        edge_1.local_actor.lr = min(1, edge_1.local_actor.lr * math.pow(1.000001, i))
        edge_1.local_critic.lr = min(1, edge_1.local_critic.lr * math.pow(1.000001, i))

        edge_2.local_actor.lr = min(1, edge_2.local_actor.lr * math.pow(1.000001, i))
        edge_2.local_critic.lr = min(1, edge_2.local_critic.lr * math.pow(1.000001, i))

        user_0.local_actor.lr = min(1, user_0.local_actor.lr * math.pow(1.000001, i))
        user_0.local_critic.lr = min(1, user_0.local_critic.lr * math.pow(1.000001, i))

        user_1.local_actor.lr = min(1, user_1.local_actor.lr * math.pow(1.000001, i))
        user_1.local_critic.lr = min(1, user_1.local_critic.lr * math.pow(1.000001, i))

        user_2.local_actor.lr = min(1, user_2.local_actor.lr * math.pow(1.000001, i))
        user_2.local_critic.lr = min(1, user_2.local_critic.lr * math.pow(1.000001, i))
        # for i in range(len(edges_g0)):
        #     edges_g0[i].local_actor.lr = min(1, edges_g0[i].local_actor.lr * math.pow(1.000001, i))
        #     edges_g0[i].local_critic.lr = min(1, edges_g0[i].local_critic.lr * math.pow(1.000001, i))
        # for i in range(len(edges_g1)):
        #     edges_g1[i].local_actor.lr = min(1, edges_g1[i].local_actor.lr * math.pow(1.000001, i))
        #     edges_g1[i].local_critic.lr = min(1, edges_g1[i].local_critic.lr * math.pow(1.000001, i))
        # for i in range(len(edges_g2)):
        #     edges_g2[i].local_actor.lr = min(1, edges_g2[i].local_actor.lr * math.pow(1.000001, i))
        #     edges_g2[i].local_critic.lr = min(1, edges_g2[i].local_critic.lr * math.pow(1.000001, i))
        # for i in range(len(edges_g3)):
        #     edges_g3[i].local_actor.lr = min(1, edges_g3[i].local_actor.lr * math.pow(1.000001, i))
        #     edges_g3[i].local_critic.lr = min(1, edges_g3[i].local_critic.lr * math.pow(1.000001, i))
        # for i in range(len(edges_g4)):
        #     edges_g4[i].local_actor.lr = min(1, edges_g4[i].local_actor.lr * math.pow(1.000001, i))
        #     edges_g4[i].local_critic.lr = min(1, edges_g4[i].local_critic.lr * math.pow(1.000001, i))
        # for i in range(len(edges_g3)):
        #     edges_g5[i].local_actor.lr = min(1, edges_g5[i].local_actor.lr * math.pow(1.000001, i))
        #     edges_g5[i].local_critic.lr = min(1, edges_g5[i].local_critic.lr * math.pow(1.000001, i))
        global GAMMA
        GAMMA = GAMMA / pow(1.00005, i)  # 1.00005
        # GAMMA = GAMMA / math.pow(1.000001, i)
        # ALPHA = ALPHA / math.pow(1.000001, i)

        # total_q_delay += q_delay_n0 #+q_d_1+q_d_2+q_d_3+q_d_4+q_d_5
        # total_drop += d_delay_n0 #d_0#+d_1+d_2+d_3+d_4+d_5
        # # total_s_delay += s_delay_n0#(r_0 - q_d_0 - 15 * d_0) #+(r_1 - q_d_1 - 15 * d_1)+(r_2 - q_d_2 - 15 * d_2)+(r_3 - q_d_3 - 15 * d_3)+\
        # #                  (r_4 - q_d_4 - 15 * d_4) + (r_5 - q_d_5 - 15 * d_5)
        # # total_utility += math.exp(-r)
        #
        # total_q_delay += q_delay_n0#/new_task_0[0] if new_task_0[0] > 0 else 0
        # total_s_delay += s_delay_n0#/new_task_0[0] if new_task_0[0] > 0 else 0
        # total_utility += math.exp(-((q_delay_n0+s_delay_n0+ d_delay_n0*15)/new_task_0[0])) if new_task_0[0] > 0 else 0 #+math.exp(-((r_1 - 15 * d_1) + 20 * (d_1 * 15)))\
        #                  # +math.exp(-((r_2 - 15 * d_2) + 20 * (d_2 * 15)))+math.exp(-((r_3 - 15 * d_3) + 20 * (d_3 * 15)))\
        #                  # +math.exp(-((r_4 - 15 * d_4) + 20 * (d_4 * 15)))+math.exp(-((r_4 - 15 * d_4) + 20 * (d_4 * 15)))
        #
        # # GAMMA = GAMMA / pow(1.0005, i_episode)
        total_r0 += r_0
        total_r1 += r_1
        total_r2 += r_2

        latency0.append(r_0)
        latency1.append(r_1)
        latency2.append(r_2)
        pp0.append(actual_p0)
        pp1.append(actual_p1)
        pp2.append(actual_p2)
        user_la_0.append(u0_q_delay)
        user_la_1.append(u1_q_delay)
        user_la_2.append(u2_q_delay)

    tf.summary.FileWriter("logs/", SESS.graph)
    tf.reset_default_graph()
    print(f"total_r0 = {total_r0} q_len0 = {q_len0} la0 = {total_r0/q_len0}", file=f)
    print(f"total_r1 = {total_r1} q_len1 = {q_len1} la1 = {total_r1/q_len1}", file=f)
    print(f"total_r2 = {total_r2} q_len2 = {q_len2} la2 = {total_r2/q_len2}", file=f)



    x = range(0, 100)
    plt.figure(1)
    plt.title('edge')
    plt.plot(x, latency0, color='#ff0000', marker='o', label='edge 0', linewidth=3.0)
    plt.plot(x, latency1, color='#00ff00', marker='o', label='edge 1', linewidth=3.0)
    plt.plot(x, latency2, color='#0000ff', marker='o', label='edge 2', linewidth=3.0)
    plt.xlabel('time', fontsize=13)
    plt.ylabel('q_delay', fontsize=13)

    plt.figure(2)
    plt.title('edge price')
    plt.plot(x, pp0, color='#FF9999', marker='o', label='p0', linewidth=3.0)
    plt.plot(x, pp1, color='#99FF99', marker='o', label='p1', linewidth=3.0)
    plt.plot(x, pp2, color='#9999FF', marker='o', label='p2', linewidth=3.0)
    plt.xlabel('time', fontsize=13)
    plt.ylabel('price', fontsize=13)

    plt.figure(3)
    plt.title('user')
    plt.xlabel('time', fontsize=13)
    plt.ylabel('q_delay', fontsize=13)
    plt.plot(x, user_la_0, color='#ff0000', marker='o', label='user 0', linewidth=3.0)
    plt.plot(x, user_la_1, color='#00ff00', marker='o', label='user 1', linewidth=3.0)
    plt.plot(x, user_la_2, color='#0000ff', marker='o', label='user 2', linewidth=3.0)

    return total_r0/q_len0, total_r1/q_len1, total_r2/q_len2, ap0, ap1, ap2, user_total_r0/user_q_len0, user_total_r1/user_q_len1, user_total_r2/user_q_len2
    # return sum(total_delay)/total_jobs, sum(total_drop)


if __name__ == "__main__":
    latency = []
    drop = []
    q_delay = []
    utility = []
    s_delay = []
    la0 = []
    la1 = []
    la2 = []
    ula0 = []
    ula1 = []
    ula2 = []
    dr = []
    # p0 = []
    # p1 = []
    # p2 = []

    l0, l1, l2, ap0, ap1, ap2, u0, u1, u2 = run(3)

    print(f'avg l0 = {l0}  l1 = {l1}  l2 = {l2}  u0 = {u0}  u1 = {u1}  u2 = {u2}', file=f)

    # x = range(1, 40, 5)
    # plt.figure(1)
    # plt.plot(x, la0, color='#ff0000', marker='o', label='edge 0', linewidth=3.0)
    # plt.plot(x, la1, color='#00ff00', marker='o', label='edge 1', linewidth=3.0)
    # plt.plot(x, la2, color='#0000ff', marker='o', label='edge 2', linewidth=3.0)
    # plt.plot(x, p0, color='#FF9999', marker='o', label='p0', linewidth=3.0)
    # plt.plot(x, p1, color='#99FF99', marker='o', label='p1', linewidth=3.0)
    # plt.plot(x, p2, color='#9999FF', marker='o', label='p2', linewidth=3.0)
    # # plt.xticks(range(35,60,5))
    # plt.xlabel('Average Task Arrivals per Slot', fontsize=13)
    # plt.ylabel('Average Queuing Delay', fontsize=13)
    # plt.tick_params(labelsize=11)
    # plt.legend(fontsize=9)
    # plt.savefig('edge.jpg')
    #
    # plt.figure(2)
    # plt.plot(x, ula0, color='#ff0000', marker='o', label='user 0', linewidth=3.0)
    # plt.plot(x, ula1, color='#00ff00', marker='o', label='user 1', linewidth=3.0)
    # plt.plot(x, ula2, color='#0000ff', marker='o', label='user 2', linewidth=3.0)
    # plt.xlabel('Average Task Arrivals per Slot', fontsize=13)
    # plt.ylabel('Average Queuing Delay', fontsize=13)
    # plt.tick_params(labelsize=11)
    # plt.legend(fontsize=9)
    # plt.savefig('user.jpg')

    plt.show()
    f.close()
