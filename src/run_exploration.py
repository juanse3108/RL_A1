from Experiment import run_exploration_experiment
import os

os.makedirs("results", exist_ok=True)

if __name__ == "__main__":
    run_exploration_experiment(output_path="results/exploration.png")