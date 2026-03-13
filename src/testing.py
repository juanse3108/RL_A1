from SARSA import SarsaAgent
import numpy as np

agent = SarsaAgent(n_states=70, n_actions=4, learning_rate=0.1, gamma=1.0)

s, a = 0, 1
s_next, a_next = 5, 2
agent.Q_sa[s_next, a_next] = 5.0

agent.update(s, a, r=-1, s_next=s_next, a_next=a_next, done=False)
print(agent.Q_sa[s,a])  # expected: 0 + 0.1 * ((-1 + 1*5) - 0) = 0.4