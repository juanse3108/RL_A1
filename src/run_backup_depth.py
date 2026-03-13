from Experiment import run_backup_depth_experiment
import os

os.makedirs("results", exist_ok=True)

if __name__ == "__main__":
    run_backup_depth_experiment(output_path="results/depth.png")