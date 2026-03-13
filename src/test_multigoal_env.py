from Environment import StochasticWindyGridworld

def test_goal_from_start(start_xy, action, expected_reward):
    env = StochasticWindyGridworld(initialize_model=False)

    # Disable wind so the transition is deterministic for this unit test
    env.wind_blows_proportion = 0.0

    env.start_location = tuple(start_xy)
    env.reset()

    s_next, r, done = env.step(action)
    print(f"Start {start_xy}, action {action} -> reward {r}, done {done}")
    assert done is True, "Should terminate when entering goal"
    assert r == expected_reward, f"Expected reward {expected_reward} but got {r}"

if __name__ == "__main__":
    # Goal [7,3]: from [6,3] go RIGHT (1)
    test_goal_from_start([6,3], 1, 100)

    # Goal [3,2]: from [2,2] go RIGHT (1)
    test_goal_from_start([2,2], 1, 5)

    print("✅ Multi-goal reward/terminal checks passed.")