#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax


class QValueIterationAgent:

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(self, s):
        ''' Returns the greedy best action in state s '''

        return np.argmax(self.Q_sa[s])

    def update(self, s, a, p_sas, r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        # TO DO: Add own code
        old_q = self.Q_sa[s, a]
        new_q = 0
        for s_next in range(len(p_sas)):
            prob = p_sas[s_next]
            reward = r_sas[s_next]
            if prob > 0:
                new_q += prob * (reward + self.gamma * np.max(self.Q_sa[s_next]))

        self.Q_sa[s, a] = new_q
        return abs(old_q - new_q)


def q_value_iteration(env, gamma=1.0, threshold=0.001):
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)
    delta = float('inf')
    iteration = 0

    while delta > threshold:
        delta = 0
        iteration += 1


        for s in range(env.n_states):
            for a in range(env.n_actions):
               p_sas, r_sas = env.model(s, a)

               error = QIagent.update(s, a, p_sas, r_sas)

               delta = max(delta, error)

        print("Iteration{},max error{}".format(iteration, delta))
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=3.0)

    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = q_value_iteration(env, gamma, threshold)

    # view optimal policy
    done = False
    s = env.reset()
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.5)
        s = s_next

def experiment(env, QIagent):
    total_reward = 0
    steps = 0
    s = env.reset()
    done = False

    while not done:
        a = QIagent.select_action(s)
        s, r, done = env.step(a)
        total_reward += r
        steps += 1
    mean_reward_per_timestep = total_reward / steps
    print("Mean reward per timestep: {}".format(mean_reward_per_timestep))



if __name__ == '__main__':
    env = StochasticWindyGridworld(initialize_model=True)
    QIagent = q_value_iteration(env, gamma=1.0, threshold=0.001)

    # Add these lines here
    start_state = env._location_to_state(np.array([0, 3]))
    V_star = np.max(QIagent.Q_sa[start_state])
    print("V*(s=3) = {}".format(V_star))

    experiment(env, QIagent)
