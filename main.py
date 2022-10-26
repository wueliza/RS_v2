import os
import random
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
from Work import MEC_network
from Edge import Edge
from User import User
import gym
import time
import random
import math
import collections

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


def run(tr):
    import time

    # State and Action Space

    N_mec_edge = 3
    N_A = N_mec_edge + 1

    total_delay = []
    total_jobs = []
    total_drop = []
    total_time = 2000
    q_len = 0
    total_r = 0
    u = 0
    total_q_delay = 0
    total_drop = 0
    total_utility = 0
    total_state_value = 0
    total_s_delay = 0
    shared_ations = np.zeros((N_mec_edge, N_mec_edge + 1))

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

    user_0 = User(scope='u' + str(0), task_arrival_rate=tr, edge_num=0, lar=0.001, lcr=0.01, q_size=50, sess=SESS)
    user_s_0 = np.hstack((user_0.q_state, user_0.CRB))  # s = [q_state, CRV]
    user_1 = User(scope='u' + str(1), task_arrival_rate=tr, edge_num=1, lar=0.001, lcr=0.01, q_size=50, sess=SESS)
    user_s_1 = np.hstack((user_1.q_state, user_1.CRB))  # s = [q_state, CRV]
    user_2 = User(scope='u' + str(2), task_arrival_rate=tr, edge_num=2, lar=0.001, lcr=0.01, q_size=50, sess=SESS)
    user_s_2 = np.hstack((user_2.q_state, user_2.CRB))  # s = [q_state, CRV]

    SESS.run(tf.global_variables_initializer())

    # store the distribution of the task to other edge
    shared_ations[0] = [0, 0, 0, 0]
    shared_ations[1] = [0, 0, 0, 0]
    shared_ations[2] = [0, 0, 0, 0]

    local_work_0 = 0
    local_work_1 = 0
    local_work_2 = 0
    local_work_type_0 = 0
    local_work_type_1 = 0
    local_work_type_2 = 0

    work_cost = [1, 2]

    # two kinds of task, [amount, cost]
    total_edge_q_len = {0: [0, 1], 1: [0, 2]}

    for i in range(total_time):
        # print("time", i, tr)

        # predict the other edge's price bace on the work distribution last time
        PD_other_price_0 = edge_0.local_predictor.choose_action(shared_ations[0]).flatten()
        PD_other_price_1 = edge_1.local_predictor.choose_action(shared_ations[1]).flatten()
        PD_other_price_2 = edge_2.local_predictor.choose_action(shared_ations[2]).flatten()
        print(f'PD_other_price_0 = {PD_other_price_0}  PD_other_price_1 = {PD_other_price_1}  PD_other_price_2 = {PD_other_price_2}')

        # merge the queue state and cpu
        s_0 = np.hstack((s_0, PD_other_price_0))
        s_1 = np.hstack((s_1, PD_other_price_1))
        s_2 = np.hstack((s_2, PD_other_price_2))
        print(f's0 = {s_0} s1 = {s_1} s2 = {s_2}')

        # actor determine the price for user of this state
        p0 = edge_0.local_actor.choose_action(s_0).flatten()[0]
        p1 = edge_1.local_actor.choose_action(s_1).flatten()[0]
        p2 = edge_2.local_actor.choose_action(s_2).flatten()[0]
        print(f'p0 = {p0}  p1 = {p1}  p2 = {p2}')

        # user
        user_s_0 = np.hstack((user_s_0, p0))
        user_s_1 = np.hstack((user_s_1, p1))
        user_s_2 = np.hstack((user_s_2, p2))

        user_0_action = user_0.local_actor.choose_action(user_s_0).flatten()[0]
        user_1_action = user_1.local_actor.choose_action(user_s_1).flatten()[0]
        user_2_action = user_2.local_actor.choose_action(user_s_2).flatten()[0]

        user_0_task, user_0_utility, user_s_0_ = user_0.step(user_0_action, p0)
        user_1_task, user_1_utility, user_s_1_ = user_0.step(user_1_action, p1)
        user_2_task, user_2_utility, user_s_2_ = user_0.step(user_2_action, p2)

        user_td_error_0 = user_0.local_critic.learn(user_s_0, user_0_utility, user_s_0_)
        user_td_error_1 = user_1.local_critic.learn(user_s_1, user_1_utility, user_s_1_)
        user_td_error_2 = user_2.local_critic.learn(user_s_2, user_2_utility, user_s_2_)

        user_0.local_actor.learn(user_s_0, user_0_action, user_td_error_0)
        user_1.local_actor.learn(user_s_1, user_1_action, user_td_error_1)
        user_2.local_actor.learn(user_s_2, user_2_action, user_td_error_2)

        # user pass the work to edge user task {type: amount}
        local_work_type_0 = list(user_0_task.items())[0][0]
        local_work_type_1 = list(user_1_task.items())[0][0]
        local_work_type_2 = list(user_2_task.items())[0][0]
        local_work_0 += list(user_0_task.items())[0][1]
        local_work_1 += list(user_1_task.items())[0][1]
        local_work_2 += list(user_2_task.items())[0][1]

        # distribute the work base on the predict price of the other edge

        shared_ations[0], actual_p0 = edge_0.distribute_work(PD_other_price_0, local_work_0, p0, work_cost[local_work_type_0])
        shared_ations[1], actual_p1 = edge_1.distribute_work(PD_other_price_1, local_work_1, p1, work_cost[local_work_type_1])
        shared_ations[2], actual_p2 = edge_2.distribute_work(PD_other_price_2, local_work_2, p2, work_cost[local_work_type_2])

        # collect the actual price of all edge
        price = [actual_p0, actual_p1, actual_p2]  # actual price

        # calculate real utility
        s_0_, total_work_0, r_0, d_0, q_d_0, avg_delay_0 = mec_0.step(shared_ations, price)  # s_, total_work_, reward, d_delay, q_delay, new_task, avg_delay
        s_1_, total_work_1, r_1, d_1, q_d_1, avg_delay_1 = mec_1.step(shared_ations, price)
        s_2_, total_work_2, r_2, d_2, q_d_2, avg_delay_2 = mec_2.step(shared_ations, price)

        if r_0 < 0 or r_1 < 0 or r_2 < 0:
            print(r_0, r_1)
            print("stop1")
            exit()

        # edge actual reward
        r_0 = r_0 + shared_ations[0][1] * avg_delay_1 + shared_ations[0][2] * avg_delay_2
        r_1 = r_1 + shared_ations[1][0] * avg_delay_0 + shared_ations[1][2] * avg_delay_2
        r_2 = r_2 + shared_ations[2][0] * avg_delay_0 + shared_ations[2][1] * avg_delay_1

        if r_0 < 0 or r_1 < 0 or r_2 < 0:
            print("stop2")
            exit()

        # total_edge_q_len[1][0] += local_work_0[1] + local_work_1[1] + local_work_2[1]
        # total_edge_q_len[2][0] += local_work_0[2] + local_work_1[2] + local_work_2[2]

        td_error_0, v_0, _, v_0_ = edge_0.local_critic.learn(p0, r_0, p0_)  # td_error = the actual price - the predict price
        td_error_1, v_1, _, v_1_ = edge_1.local_critic.learn(p1, r_1, p1_)
        td_error_2, v_2, _, v_2_ = edge_1.local_critic.learn(p2, r_2, p2_)

        edge_0.local_actor.learn(s_0, p0, td_error_0)
        edge_1.local_actor.learn(s_1, p1, td_error_1)
        edge_2.local_actor.learn(s_2, p2, td_error_2)

        edge_0.local_predictor.learn(PD_other_price_0, price, avg_delay_0)
        edge_1.local_predictor.learn(PD_other_price_1, price, avg_delay_1)
        edge_2.local_predictor.learn(PD_other_price_2, price, avg_delay_2)

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
        total_r += r_0 + r_1 + r_2  # + r_1+r_2+r_3+r_4+r_5

    # print("GAMMA", GAMMA)
    print("task", q_len)
    # print(total_jobs)
    # print("drop", total_drop/total_time)
    # print(total_delay)
    # try:
    #     latency[tr] = sum(total_delay)/total_jobs
    # except:
    #     print(tr)
    tf.summary.FileWriter("logs/", SESS.graph)
    tf.reset_default_graph()
    return total_r / q_len, total_drop / total_time, total_q_delay, total_utility, total_s_delay
    # return sum(total_delay)/total_jobs, sum(total_drop)


if __name__ == "__main__":
    latency = []
    drop = []
    q_delay = []
    utility = []
    s_delay = []
    la = []
    dr = []

    for j in range(5):
        latency = []
        drop = []
        q_delay = []
        utility = []
        s_delay = []
        for i in range(1, 40):  # task arrival rate
            print(j, i)
            # i,r =pool.apply_async(func=run, args=(i,))
            # print((i,r))
            l, d, l_q, u, l_s = run(i)
            # ans.append(l)
            # drop.append(d)
            # p.start()
            # pros.append(p)
            latency.append(l)
            print(latency)
            drop.append(d)
            q_delay.append(l_q)
            utility.append(u)
            s_delay.append(l_s)
        la.append(latency)
        dr.append(drop)
    print(la)
    la = np.mean(np.array(la), axis=0)
    dr = np.mean(np.array(dr), axis=0)
    print(la)
    # la = np.mean(a, axis=0)

    ############### Measure performance only#############
    # import multiprocessing  as mp
    # import os
    # for rate in range(5,85,5):
    #     s = time.time()
    #     p = mp.Process(target=run, args=(rate,))
    #     p.start()
    #     print(p.pid)
    #     file = "ac_"+str(N_mec_edge)+"n_r"+str(rate)+"_activity_dis.txt"
    #     print(file)
    #     command = "psrecord "+str(p.pid)+ " --log /Users/hsiehli-tse/Desktop/Reseearch_2020_3/MEC_performance/"+ file+\
    #               " --duration 150 --interval 1 --include-children"
    #     print(command)
    #     os.system(command)
    #     os.system("kill -9 "+ str(p.pid))
    #     print(time.time()-s)

import pickle
#
# with open(r"/Users/hsiehli-tse/Downloads/Reinforcement-learning-with-tensorflow-master/contents/8_Actor_Critic_Advantage/ac_dis_2n_l_v7.txt","wb") as fp:
#     pickle.dump(la, fp)
# with open(r"/Users/hsiehli-tse/Downloads/Reinforcement-learning-with-tensorflow-master/contents/8_Actor_Critic_Advantage/ac_dis_2n_q_v7.txt","wb") as fp:
#     pickle.dump(q_delay, fp)
# with open(r"/Users/hsiehli-tse/Downloads/Reinforcement-learning-with-tensorflow-master/contents/8_Actor_Critic_Advantage/ac_dis_2n_d_v7.txt","wb") as fp:
#     pickle.dump(dr, fp)
# with open(r"/Users/hsiehli-tse/Downloads/Reinforcement-learning-with-tensorflow-master/contents/8_Actor_Critic_Advantage/ac_dis_2n_s_v7.txt","wb") as fp:
#     pickle.dump(s_delay, fp)
# with open(r"/Users/hsiehli-tse/Downloads/Reinforcement-learning-with-tensorflow-master/contents/8_Actor_Critic_Advantage/ac_dis_2n_u_v7.txt","wb") as fp:
#     pickle.dump(utility, fp)

import matplotlib.pyplot as plt

x = range(1, 40)
plt.plot(x, la, color='#9D2EC5', marker='o', label='Distributed Actor Critic', linewidth=3.0)
# plt.plot(x, ac_12n_dis_l ,color= '#F5B14C',marker='o',label='Distributed Actor Critic (group = 1)', linewidth=3.0)
plt.title("add_ActorLearning")

# plt.xticks(range(35,60,5))
plt.xlabel('Average Task Arrivals per Slot', fontsize=13)
plt.ylabel('Average Service Delay', fontsize=13)
plt.tick_params(labelsize=11)
plt.legend(fontsize=9)
plt.show()
