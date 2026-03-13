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
        # TO DO: Add own code ✅

        current_q = self.Q_sa[s,a]

        if done == True:
            target = 0
        else:
            target = np.max(self.Q_sa[s_next,:])

        tunning = self.learning_rate * ( (r) + (self.gamma * target) - (current_q) )

        self.Q_sa[s,a] = current_q + tunning

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)

    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)

    eval_timesteps = []
    eval_returns = []
    
    # TO DO: Write your Q-learning algorithm here! ✅
    
    s = env.reset()

    for t in range(n_timesteps):
        # we evaluate first with our current Q
        if t % eval_interval == 0:
             mean_return = agent.evaluate(eval_env)
             eval_returns.append(mean_return)
             eval_timesteps.append(t)

        # select action for our current state and selected policy
        a = agent.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
        # we take the step and calculate the next one, reward of the step and if its terminal
        s_next, r, done = env.step(a)
        # we update our Q from our agent with our tailored-made uodate function
        agent.update(s, a, r, s_next, done)
        if done == True:
             s = env.reset()
        else:
             s = s_next

             

    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution

    if plot:
        env.render(Q_sa=agent.Q_sa, plot_optimal_policy=True, step_pause=0.1)
    
    return np.array(eval_returns), np.array(eval_timesteps)

##############################################################################################################################
def evaluate_with_goal_counts(agent, eval_env, n_eval_episodes=100, max_episode_length=100):
    goal_counts = {0: 0, 1: 0, "none": 0}
    returns = []

    for _ in range(n_eval_episodes):
        s = eval_env.reset()
        R_ep = 0
        reached_goal_idx = None

        for _ in range(max_episode_length):
            a = agent.select_action(s, policy="greedy")
            s_next, r, done = eval_env.step(a)
            R_ep += r
            s = s_next
            if done:
                loc = eval_env.state_to_location(s)
                for i, g in enumerate(eval_env.goal_locations):
                    if [loc[0], loc[1]] == g:
                        reached_goal_idx = i
                        break
                break

        returns.append(R_ep)
        if reached_goal_idx is None:
            goal_counts["none"] += 1
        else:
            goal_counts[reached_goal_idx] += 1

    return float(np.mean(returns)), goal_counts

def test():
    
    n_timesteps = 500
    eval_interval=100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
    print(eval_returns,eval_timesteps)

    import matplotlib.pyplot as plt
    plt.show()

##############################################################################################################################
### for 1.5.d
# def test():
#     n_timesteps = 100000
#     gamma = 1.0
#     learning_rate = 0.1

#     policy = "softmax"
#     epsilon = 0.3
#     temp = 0.1
#     plot = False

#     # Train and keep the trained agent (do it here explicitly)
#     env = StochasticWindyGridworld(initialize_model=False)
#     agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)

#     s = env.reset()
#     for _ in range(n_timesteps):
#         a = agent.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
#         s_next, r, done = env.step(a)
#         agent.update(s, a, r, s_next, done)
#         s = env.reset() if done else s_next

#     # Evaluate and count goals
#     eval_env = StochasticWindyGridworld(initialize_model=False)
#     mean_return, goal_counts = evaluate_with_goal_counts(agent, eval_env, n_eval_episodes=200)

#     print("Mean greedy eval return:", mean_return)
#     print("Goal counts:", goal_counts)

if __name__ == '__main__':
    test()