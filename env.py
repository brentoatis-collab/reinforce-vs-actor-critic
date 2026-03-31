import numpy as np


class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles = {(1, 1), (2, 2)}
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        row, col = self.state
        d_row, d_col = self.actions[action]

        next_row = max(0, min(self.size - 1, row + d_row))
        next_col = max(0, min(self.size - 1, col + d_col))
        next_state = (next_row, next_col)

        if next_state in self.obstacles:
            self.state = next_state
            return next_state, -10, True

        if next_state == self.goal:
            self.state = next_state
            return next_state, 10, True

        self.state = next_state
        return next_state, -1, False

    def is_terminal(self):
        return self.state == self.goal or self.state in self.obstacles


# 🔥 TEST BLOCK
if __name__ == "__main__":
    env = GridWorld()
    state = env.reset()
    actions_map = ["UP", "DOWN", "LEFT", "RIGHT"]
    print("Initial State:", state)

    for _ in range(5):
        action = np.random.choice(4)
        next_state, reward, done = env.step(action)
        print(f"Action: {actions_map[action]} -> Next: {next_state}, Reward: {reward}, Done: {done}")
        if done:
            if reward == 10:
                print("Goal reached. Episode complete.")
            elif reward == -10:
                print("Hit an obstacle. Episode terminated.")
            break