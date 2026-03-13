#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class SarsaAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,a_next,done):
        # TO DO: Add own code
        current_q = self.Q_sa[s,a]

        if done == True:
            target = 0
        else:
            target = self.Q_sa[s_next,a_next]

        tunning = self.learning_rate * ( (r) + (self.gamma * target) - (current_q) )

        self.Q_sa[s,a] = current_q + tunning

        
def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)

    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)

    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your SARSA algorithm here!

    s = env.reset()
    a = pi.select_action(s,policy=policy,epsilon=epsilon,temp=temp)

    for t in range(n_timesteps):
        #first we evaluate every number of timesteps
        if t % eval_interval == 0:
            mean_return = pi.evaluate(eval_env)
            eval_returns.append(mean_return)
            eval_timesteps.append(t)
        
        #we TAKE THE STEP and calculate next step, reward and done status
        s_next, r, done = env.step(a)
        a_next = pi.select_action(s_next,policy=policy,epsilon=epsilon,temp=temp)
        #we UPDATE our Q matrix with the current info according to SARSA
        pi.update(s,a,r,s_next,a_next,done)
        
        if done:
            s = env.reset()
        else:
            s,a = s_next,a_next

    if plot:
       env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during SARSA execution

    return np.array(eval_returns), np.array(eval_timesteps) 


def test():
    n_timesteps = 1000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
            
# def test():
#     n_timesteps = 20000
#     eval_interval = 100
#     gamma = 1.0
#     learning_rate = 0.1

#     # Exploration settings
#     policy = 'egreedy'
#     epsilon = 0.1
#     temp = 1.0

#     plot = False

#     eval_returns, eval_timesteps = sarsa(
#         n_timesteps,
#         learning_rate,
#         gamma,
#         policy,
#         epsilon,
#         temp,
#         plot,
#         eval_interval
#     )

#     print("Eval returns:", eval_returns)
#     print("Eval timesteps:", eval_timesteps)
#     print("Final return:", eval_returns[-1])

if __name__ == '__main__':
    test()
