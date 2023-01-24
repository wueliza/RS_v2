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
    user_s_0 = np.hstack((user_0.q_state, user_0.CRB))  # s = [q_state, CRB]
    user_1 = User(scope='u' + str(1), task_arrival_rate=tr, edge_num=1, lar=0.001, lcr=0.01, q_size=50, sess=SESS)
    user_s_1 = np.hstack((user_1.q_state, user_1.CRB))  # s = [q_state, CRB]
    user_2 = User(scope='u' + str(2), task_arrival_rate=tr, edge_num=2, lar=0.001, lcr=0.01, q_size=50, sess=SESS)
    user_s_2 = np.hstack((user_2.q_state, user_2.CRB))  # s = [q_state, CRB]

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
    ap0 = []
    ap1 = []
    ap2 = []

    # q_len0 += s_0[0]
    # q_len1 += s_1[0]
    # q_len2 += s_2[0]
    # print(f'\nq0 = {q_len0}  q1 = {q_len1}  q2 = {q_len2}', file=f)

    for i in range(total_time):
        print("\ntime = ", i, file=f)
        # q_len0 += s_0[0]
        # q_len1 += s_1[0]
        # q_len2 += s_2[0]
        # print(f'\nq0 = {q_len0}  q1 = {q_len1}  q2 = {q_len2}', file=f)

        # predict the other edge's price bace on the work distribution last time
        PD_other_price_0 = edge_0.local_predictor.choose_action(shared_ations[0]).flatten()
        PD_other_price_1 = edge_1.local_predictor.choose_action(shared_ations[1]).flatten()
        PD_other_price_2 = edge_2.local_predictor.choose_action(shared_ations[2]).flatten()
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
        user_s_0 = np.hstack((user_s_0, p0))
        user_s_1 = np.hstack((user_s_1, p1))
        user_s_2 = np.hstack((user_s_2, p2))
        print(f'user_s0 = {user_s_0}  user_s1 = {user_s_1}  user_s2 = {user_s_2}', file=f)

        user_0_action = user_0.local_actor.choose_action(user_s_0).flatten()[0]
        user_1_action = user_1.local_actor.choose_action(user_s_1).flatten()[0]
        user_2_action = user_2.local_actor.choose_action(user_s_2).flatten()[0]
        print(f'user0 action = {user_0_action}  user1 action = {user_1_action}  user2 action = {user_2_action}', file=f)

        user_0_task, user_0_utility, user_s_0_, u0_tr, u0_work, u0_overflow, u0_q_delay = user_0.step(user_0_action, p0)
        user_1_task, user_1_utility, user_s_1_, u1_tr, u1_work, u1_overflow, u1_q_delay = user_0.step(user_1_action, p1)
        user_2_task, user_2_utility, user_s_2_, u2_tr, u2_work, u2_overflow, u2_q_delay = user_0.step(user_2_action, p2)
        print(f'user0 trans task = {user_0_task}  utility = {user_0_utility}  user_s_ = {user_s_0_}  tr = {u0_tr}  work = {u0_work}  overflow = {u0_overflow}  q_delay = {u0_q_delay}', file=f)
        print(f'user1 trans task = {user_1_task}  utility = {user_1_utility}  user_s_ = {user_s_1_}  tr = {u1_tr}  work = {u1_work}  overflow = {u1_overflow}  q_delay = {u1_q_delay}', file=f)
        print(f'user2 trans task = {user_2_task}  utility = {user_2_utility}  user_s_ = {user_s_2_}  tr = {u2_tr}  work = {u2_work}  overflow = {u2_overflow}  q_delay = {u2_q_delay}', file=f)

        user_td_error_0 = user_0.local_critic.learn(user_s_0, user_0_utility, user_s_0_)
        user_td_error_1 = user_1.local_critic.learn(user_s_1, user_1_utility, user_s_1_)
        user_td_error_2 = user_2.local_critic.learn(user_s_2, user_2_utility, user_s_2_)

        user_0.local_actor.learn(user_s_0, user_0_action, user_td_error_0)
        user_1.local_actor.learn(user_s_1, user_1_action, user_td_error_1)
        user_2.local_actor.learn(user_s_2, user_2_action, user_td_error_2)

        user_s_0 = user_s_0_[:len(user_s_0_)-1]
        user_s_1 = user_s_1_[:len(user_s_1_)-1]
        user_s_2 = user_s_2_[:len(user_s_2_)-1]
        # user pass the work to edge user task {type: amount}
        local_work_type_0 = list(user_0_task.items())[0][0]
        local_work_type_1 = list(user_1_task.items())[0][0]
        local_work_type_2 = list(user_2_task.items())[0][0]
        local_work_0 += list(user_0_task.items())[0][1]
        local_work_1 += list(user_1_task.items())[0][1]
        local_work_2 += list(user_2_task.items())[0][1]
        print(f'local work 0 = {local_work_0}  local work 1 = {local_work_1}  local work 2 = {local_work_2}', file=f)

        # distribute the work base on the predict price of the other edge

        shared_ations[0], actual_p0 = edge_0.distribute_work(PD_other_price_0, local_work_0, p0, work_cost[local_work_type_0])
        shared_ations[1], actual_p1 = edge_1.distribute_work(PD_other_price_1, local_work_1, p1, work_cost[local_work_type_1])
        shared_ations[2], actual_p2 = edge_2.distribute_work(PD_other_price_2, local_work_2, p2, work_cost[local_work_type_2])
        print(f'actual price p0 = {actual_p0}  p1 = {actual_p1}  p2 = {actual_p2} \nshared action: ',file=f)
        print(shared_ations, file=f)

        # collect the actual price of all edge
        price = [actual_p0, actual_p1, actual_p2, COST_TO_CLOUD]  # actual price
        ap0.append(actual_p0)
        ap1.append(actual_p1)
        ap2.append(actual_p2)
        work_type = [local_work_type_0, local_work_type_1, local_work_type_2]
        print(f'work type = {work_type}', file=f)
        # calculate real utility

        s_0_, total_work_0, r_0, d_0, q_d_0, avg_delay_0, paid_0, lj0, tj0, overflow0 = mec_0.step(shared_ations, price, work_type)  # s_, total_work_, reward, d_delay, q_delay, new_task, avg_delay
        s_1_, total_work_1, r_1, d_1, q_d_1, avg_delay_1, paid_1, lj1, tj1, overflow1 = mec_1.step(shared_ations, price, work_type)
        s_2_, total_work_2, r_2, d_2, q_d_2, avg_delay_2, paid_2, lj2, tj2, overflow2 = mec_2.step(shared_ations, price, work_type)

        q_len0 += total_work_0
        q_len1 += total_work_1
        q_len2 += total_work_2
        print(f'q0 = {q_len0}  q1 = {q_len1}  q2 = {q_len2}', file=f)
        print(f'edge0: s = {s_0_} work = {total_work_0} r = {r_0} d = {d_0} qd = {q_d_0} ad = {avg_delay_0} paid = {paid_0} local_job = {lj0} total_job = {tj0} overflow = {overflow0}', file=f)
        print(f'edge1: s = {s_1_} work = {total_work_1} r = {r_1} d = {d_1} qd = {q_d_1} ad = {avg_delay_1} paid = {paid_1} local_job = {lj1} total_job = {tj1} overflow = {overflow1}', file=f)
        print(f'edge2: s = {s_2_} work = {total_work_2} r = {r_2} d = {d_2} qd = {q_d_2} ad = {avg_delay_2} paid = {paid_2} local_job = {lj2} total_job = {tj2} overflow = {overflow2}', file=f)

        # if r_0 < 0 or r_1 < 0 or r_2 < 0:
        #     print(r_0, r_1)
        #     print("stop1")
        #     exit()

        # edge actual reward
        r_0 = r_0 - user_0_utility
        r_1 = r_1 - user_1_utility
        r_2 = r_2 - user_2_utility
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
        edge_1.local_predictor.learn(PD_other_price_1, r_1, price)
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

    tf.summary.FileWriter("logs/", SESS.graph)
    tf.reset_default_graph()
    print(f"total_r0 = {total_r0} q_len0 = {q_len0} la0 = {total_r0/q_len0}", file=f)
    print(f"total_r1 = {total_r1} q_len1 = {q_len1} la1 = {total_r1/q_len1}", file=f)
    print(f"total_r2 = {total_r2} q_len2 = {q_len2} la2 = {total_r2/q_len2}", file=f)
    return total_r0/q_len0, total_r1/q_len1, total_r2/q_len2, ap0, ap1, ap2
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
    dr = []
    p0 = []
    p1 = []
    p2 = []

    for j in range(5):
        latency0 = []
        latency1 = []
        latency2 = []
        pp0 = []
        pp1 = []
        pp2 = []

        for i in range(1, 40, 5):  # task arrival rate
            print(j, i)
            print('\n', j, i, file=f)
            # i,r =pool.apply_async(func=run, args=(i,))
            # print((i,r))
            l0, l1, l2, ap0, ap1, ap2 = run(i)

            latency0.append(l0)
            latency1.append(l1)
            latency2.append(l2)
            pp0.append(np.mean(np.array(ap0), axis=0))
            pp1.append(np.mean(np.array(ap1), axis=0))
            pp2.append(np.mean(np.array(ap2), axis=0))

        la0.append(latency0)
        la1.append(latency1)
        la2.append(latency2)
        p0.append(pp0)
        p1.append(pp1)
        p2.append(pp2)
        print('la0 = ', end='', file=f)
        print(la0, file=f)
        print('la1 = ', end='', file=f)
        print(la1, file=f)
        print('la2 = ', end='', file=f)
        print(la2, file=f)

    la0 = np.mean(np.array(la0), axis=0)
    la1 = np.mean(np.array(la1), axis=0)
    la2 = np.mean(np.array(la2), axis=0)
    p0 = np.mean(np.array(p0), axis=0)
    p1 = np.mean(np.array(p1), axis=0)
    p2 = np.mean(np.array(p2), axis=0)
    print('la0 = ', end='', file=f)
    print(la0, file=f)
    print('la1 = ', end='', file=f)
    print(la1, file=f)
    print('la2 = ', end='', file=f)
    print(la2, file=f)
    print('p0 = ', end='', file=f)
    print(p0, file=f)
    print('p1 = ', end='', file=f)
    print(p1, file=f)
    print('p2 = ', end='', file=f)
    print(p2, file=f)
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

    x = range(1, 40, 5)
    plt.plot(x, la0, color='#ff0000', marker='o', label='edge 0', linewidth=3.0)
    plt.plot(x, la1, color='#00ff00', marker='o', label='edge 1', linewidth=3.0)
    plt.plot(x, la2, color='#0000ff', marker='o', label='edge 2', linewidth=3.0)
    plt.plot(x, p0, color='#FF9999', marker='o', label='p0', linewidth=3.0)
    plt.plot(x, p1, color='#99FF99', marker='o', label='p1', linewidth=3.0)
    plt.plot(x, p2, color='#9999FF', marker='o', label='p2', linewidth=3.0)
    # plt.xticks(range(35,60,5))
    plt.xlabel('Average Task Arrivals per Slot', fontsize=13)
    plt.ylabel('Average Reward/Price', fontsize=13)
    plt.tick_params(labelsize=11)
    plt.legend(fontsize=9)
    plt.show()
    f.close()
