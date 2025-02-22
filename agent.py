import numpy as np
import random
import pickle  # For saving and loading Q-table


class QLearningAgent:
    """
    A Q-learning agent that interacts with the MazeEnv environment.
    """

    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initializes the Q-learning agent.

        :param env: The maze environment
        :param alpha: Learning rate
        :param gamma: Discount factor
        :param epsilon: Exploration probability
        """
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration probability

        # Q-table initialized to zeros: state (x, y) -> action values
        self.q_table = {}
        for x in range(env.grid_size[0]):
            for y in range(env.grid_size[1]):
                self.q_table[(x, y)] = {action: 0.0 for action in env.get_action_space()}

    def choose_action(self, state):
        """
        Selects an action using the epsilon-greedy strategy.

        :param state: Current state (x, y)
        :return: Selected action
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.get_action_space())  # Explore
        else:
            return max(self.q_table[state], key=self.q_table[state].get)  # Exploit

    def update_q_value(self, state, action, reward, next_state, done):
        """
        Updates the Q-table using the Bellman equation.

        :param state: Current state (x, y)
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state (x, y)
        :param done: Boolean indicating if the episode is over
        """
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        target = reward + (0 if done else self.gamma * self.q_table[next_state][best_next_action])
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    def save_q_table(self, filename="models/q_table.pkl"):
        """
        Saves the Q-table to a file.
        """
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename="models/q_table.pkl"):
        """
        Loads the Q-table from a file.
        """
        try:
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print("No saved Q-table found. Training from scratch.")
