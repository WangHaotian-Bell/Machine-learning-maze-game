import numpy as np


class MazeEnv:
    def __init__(self, grid_size=(10, 10), start=(0, 0), goal=(9, 9), obstacles=None):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.state = start  # Current position of the agent
        self.action_space = ['up', 'down', 'left', 'right']
        self.num_actions = len(self.action_space)

        # Initialize the grid and set obstacles
        self.obstacles = obstacles if obstacles else []
        self.grid = np.zeros(grid_size)
        for obs in self.obstacles:
            self.grid[obs] = -1  # Mark obstacles

        self.grid[goal] = 1  # Goal position has a positive reward

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 'up':
            new_state = (max(x - 1, 0), y)
        elif action == 'down':
            new_state = (min(x + 1, self.grid_size[0] - 1), y)
        elif action == 'left':
            new_state = (x, max(y - 1, 0))
        elif action == 'right':
            new_state = (x, min(y + 1, self.grid_size[1] - 1))
        else:
            raise ValueError("Invalid action.")

        # Check if new state is an obstacle
        if new_state in self.obstacles:
            new_state = self.state
            reward = -1
            done = False  # Ensure 'done' is explicitly set
        elif new_state == self.goal:
            reward = 10
            done = True  # Ensure episode ends at the goal
        else:
            reward = -0.1
            done = False

        self.state = new_state
        return new_state, reward, done

    def render(self):
        grid_display = np.copy(self.grid)
        x, y = self.state
        grid_display[x, y] = 2  # Represent agent as '2'

        print("\nMaze Grid:")
        for row in grid_display:
            print(" ".join(["A" if cell == 2 else "X" if cell == -1 else "G" if cell == 1 else "." for cell in row]))
        print()

    def get_action_space(self):
        return self.action_space
