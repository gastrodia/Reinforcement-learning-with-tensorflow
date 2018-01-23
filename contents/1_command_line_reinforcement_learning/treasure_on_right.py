#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import print_function

'''
让计算机自己学习

老师 只对结果打分

强化学习具有分数导向性
可以一次次在环境中的尝试 获取分数标签


监督学习是数据导向

真实反馈 vs 现实世界建模 想象力

'''


"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible

'''
可以建立一个模型来模拟机器的反馈
'''

N_STATES = 6   # the length of the 1 dimensional world 问题的难度
ACTIONS = ['left', 'right']     # available actions 可能的动作
EPSILON = 0.9   # greedy police  有多少不变 
ALPHA = 0.1     # learning rate 学习效率
GAMMA = 0.9    # discount factor  衰减  眼镜比喻 越接近1 机器人越有远见
MAX_EPISODES = 13   # maximum episodes 最大探索步骤
FRESH_TIME = 0.3    # fresh time for one move 延时 为了看动画效果

#创建q_table
# 为啥叫Ｑ表 query的意思么？
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name  用actions的name 当表的列
    )
    # print(table)    # show table
    return table

#
def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    #行为随机变化
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name

#从环境中获取反馈
def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':    # move right  当前在向右走的话才有可能
        if S == N_STATES - 2:   # terminate  所在的位Ｓ- 2,答案的前一个位？
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left  南苑北泽
        R = 0
        if S == 0:  #移动到最前面的了，不用移动了
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter): #重绘画面
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


'''
对得分的理解是最有意思的
什么是预计得分
什么是实际得分

想象的或者说是经验的：就是Ｑ表里的值咯

啥是现实？
现实也是一种估计，一个很有遇见的估计！
现实的得分 = 真的把事干了后的得分 + 接下来可能获得的奖励 （即：下一步最大的得分可能 * 预见能力）

接下来可能获得的奖励 其实也是根据经验（Ｑ表）算的

Zdzisław Pawlak
不愧是搞出了模糊集理论的波兰大佬


'''

def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES): #
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)  #更新环境 重绘画面尔已不用太在意
        while not is_terminated:

            A = choose_action(S, q_table)
            q_predict = q_table.loc[S, A]  #估计一下这一步能得多少分

            #真的去干了哦
            #获得环境的反馈，下一个状态和奖励
            #走对的路 获得宝藏才能得1  否则都只能得0  这里的得分是这一步的实际得分
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward 
            

            #不过孙子兵法说 凡事不可只看其形  还要看其势
            #就好比你写了牛逼的算法去赚钱 ，赚了点钱，然后跑来了个投资人，要给你投资、给你的项目定价（其实也是一种估值）
            #这时候衡量你算法的价值就是你当前赚的钱么，不要这么天真
            #至少还要考虑下这个算法未来还能赚多少钱啊 即便你经验有限考虑不到太远的未来 也至少想下下个月能赚多少钱嘛
            #算不准也没关系 你只要积累经验、 因为你的算法太牛逼了 导致这个投资人天天来找你
            #而且还每次来都给你重新估值重新投你的机会
 
            if S_ != 'terminal': #根据q表 获得最大值
                #q现实 就是不太明白为啥这个地方叫现实 
                print('q_table.iloc[S_, :]') 
                print(q_table.iloc[S_, :])
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal 其实是想象
                # 实际得分 = 当前的奖励 + 接下来可能获得的奖励
                # 接下来可能获得的奖励 = 获得奖励的最大可能性 * 眼镜的度数
            else:#是最后的话 就不用估计下一步啦
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            #估计和现实的差距
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update   更新Q表
            S = S_  # move to next state

            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
