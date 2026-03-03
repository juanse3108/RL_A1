#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for master course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Helper import softmax, argmax

class BaseAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        if policy == 'greedy':
            # TO DO: Add own code ✅
            #a = np.random.randint(0,self.n_actions) # Replace this with correct action selection

            # greedy takes the action which gives the optimal reward from Q hence max argument
            a = argmax(self.Q_sa[s,:])
            
        elif policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
                
            # TO DO: Add own code ✅
            #a = np.random.randint(0,self.n_actions) # Replace this with correct action selection

            # egreedy gives random action with p = epsilon, otherwise greedy option
            greedy_choice = argmax(self.Q_sa[s,:])
            random_choice = np.random.randint(0, self.n_actions)

            a = np.random.choice([greedy_choice,random_choice],p=[1-epsilon,epsilon])
                 
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
                
            # TO DO: Add own code ✅
            #a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
  
            probs = softmax(self.Q_sa[s,:],temp)

            a = np.random.choice(range(self.n_actions),p=probs)


        return a
        
    def update(self):
        raise NotImplementedError('For each agent you need to implement its specific back-up method') # Leave this and overwrite in subclasses in other files


    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = self.select_action(s, 'greedy')
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return
