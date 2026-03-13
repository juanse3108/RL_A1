from Experiment import run_exploration_experiment
import os

os.makedirs("results", exist_ok=True)

if __name__ == "__main__":
    run_exploration_experiment(
        output_path="results/exploration_multiple_goal.png",
        env_kwargs={
            "goal_locations": [[7, 3], [3, 2]],
            "goal_rewards": [100, 5],
        }
    )