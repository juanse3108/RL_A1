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

class QLearningAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,done):
        if done:
            target = r #no more moves
        elif not done:
            target = r + self.gamma * np.max(self.Q_sa[s_next]) #page 6, figure 4

        self.Q_sa[s, a] += self.learning_rate * (target - self.Q_sa[s, a])

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=False, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    
    # TO DO: Write your Q-learning algorithm here! 
    #algorithm 2, page 8
    s = env.reset()

    for t in range(n_timesteps):

        #evaluate
        if t % eval_interval == 0:
            eval_return = agent.evaluate(eval_env)
            
            #stores eval and time
            eval_returns.append(eval_return)
            eval_timesteps.append(t)

        a = agent.select_action(s, policy, epsilon, temp)

        #new step 
        s_next, r, done = env.step(a)

        #new step, new update
        agent.update(s, a, r, s_next, done)

        if done:
            s = env.reset()
        else:
            s = s_next
    
        if plot:
            env.render(Q_sa=agent.Q_sa, plot_optimal_policy=True, step_pause=0.1) # Plot the Q-value estimates during Q-learning execution
    
    # eval_return = agent.evaluate(eval_env)
    # eval_returns.append(eval_return)
    # eval_timesteps.append(n_timesteps)
    
    return np.array(eval_returns), np.array(eval_timesteps)   

def test():
    
    n_timesteps = 1000
    eval_interval=100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = False

    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
    print(eval_returns,eval_timesteps)

if __name__ == '__main__':
    test()
