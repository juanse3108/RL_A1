# Reinforcement Learning Assignment 1  
Leiden University – MSc Computer Science  
Course: Reinforcement Learning

This repository contains the implementation and experiments for Assignment 1.  
The project includes implementations of:

- Dynamic Programming (Q-value iteration)
- Q-learning
- SARSA
- n-step Q-learning
- Monte Carlo control

Experiments reproduce the analyses described in the report.

---

# Project Structure

RL_A1/
│
├── src/                # Source code
│   ├── run_dp.py
│   ├── run_exploration.py
│   ├── run_exploration_multigoal.py
│   ├── run_on_off_policy.py
│   ├── run_backup_depth.py
│   └── …
│
├── results/            # Generated plots
├── docs/               # Report and figures
├── requirements.txt
└── README.md

---

# Installation

Create a Python environment and install dependencies:

pip install -r requirements.txt

---

# Running Experiments

Each experiment can be reproduced using a **single command** as required by the assignment.

Run commands from the project root (`RL_A1/`).

## 1. Dynamic Programming

python src/run_dp.py

Runs Q-value iteration and prints the optimal value and evaluation statistics.

---

## 2. Exploration Strategies

python src/run_exploration.py

Compares:

- ε-greedy exploration
- softmax exploration

Output:

results/exploration.png

---

## 3. Exploration with Multiple Goals

python src/run_exploration_multigoal.py

Evaluates exploration behaviour when the environment contains two goals with different rewards.

Output:

results/exploration_multiple_goal.png

---

## 4. Q-learning vs SARSA (On-policy vs Off-policy)

python src/run_on_off_policy.py

Output:

results/on_off_policy.png

---

## 5. Backup Depth Experiments

python src/run_backup_depth.py

Compares:

- n-step Q-learning for different values of *n*
- Monte Carlo control

Output:

results/depth.png

---

# Notes

- Experiments average results over multiple repetitions.
- Plots are saved automatically in the `results/` directory.
- The stochastic Windy Gridworld environment is provided in `Environment.py`.

---

# Authors

Juan Sebastián Meléndez Granados  - MSc Computer Science
Emmanuel Oblitey Ashong  - MSc Computer Science
Theodoros-Marios Adalakis  - MSc Computer Science
Leiden University