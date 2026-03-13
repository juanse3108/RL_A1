from Experiment import run_on_off_policy_experiment
import os

os.makedirs("results", exist_ok=True)

if __name__ == "__main__":
    run_on_off_policy_experiment(output_path="results/on_off_policy.png")