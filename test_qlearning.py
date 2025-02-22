import unittest
import numpy as np
from environment import MazeEnv
from agent import QLearningAgent

class TestMazeEnv(unittest.TestCase):
    """Tests for the Maze environment."""

    def setUp(self):
        """Set up a small 5x5 test maze with obstacles."""
        self.env = MazeEnv(grid_size=(5, 5), start=(0, 0), goal=(4, 4), obstacles=[(2, 2), (3, 3)])

    def test_initial_state(self):
        """Check if the environment initializes correctly."""
        self.assertEqual(self.env.state, (0, 0))

    def test_step_movement(self):
        """Check if the agent moves correctly."""
        next_state, reward, done = self.env.step("right")
        self.assertEqual(next_state, (0, 1))  # Move right
        self.assertFalse(done)

    def test_step_boundary(self):
        """Ensure the agent doesn't move out of bounds."""
        self.env.state = (0, 0)
        next_state, _, _ = self.env.step("up")
        self.assertEqual(next_state, (0, 0))  # Should stay in place

    def test_step_obstacle(self):
        """Ensure the agent doesn't move into obstacles."""
        self.env.state = (1, 2)  # Left of obstacle (2,2)
        next_state, reward, _ = self.env.step("down")
        self.assertEqual(next_state, (1, 2))  # Should remain in place
        self.assertEqual(reward, -1)  # Obstacle penalty

    def test_goal_reached(self):
        """Ensure reaching the goal ends the episode."""
        self.env.state = (3, 4)
        _, reward, done = self.env.step("down")
        self.assertTrue(done)
        self.assertEqual(reward, 10)  # Goal reward

class TestQLearningAgent(unittest.TestCase):
    """Tests for the Q-learning agent."""

    def setUp(self):
        """Set up a test agent in a simple 5x5 maze."""
        self.env = MazeEnv(grid_size=(5, 5), start=(0, 0), goal=(4, 4))
        self.agent = QLearningAgent(self.env, alpha=0.1, gamma=0.9, epsilon=0.1)

    def test_q_table_initialization(self):
        """Check if the Q-table is initialized correctly."""
        for state in self.agent.q_table:
            self.assertTrue(all(a in self.agent.q_table[state] for a in self.env.get_action_space()))
            self.assertTrue(all(self.agent.q_table[state][a] == 0.0 for a in self.env.get_action_space()))

    def test_choose_action_exploration(self):
        """Ensure the agent chooses actions correctly using epsilon-greedy."""
        state = (0, 0)
        actions = [self.agent.choose_action(state) for _ in range(100)]
        self.assertTrue(any(a in self.env.get_action_space() for a in actions))  # Valid actions
        self.assertGreater(len(set(actions)), 1)  # Some randomness due to epsilon

    def test_q_value_update(self):
        """Check if the Q-value updates correctly using the Bellman equation."""
        state = (0, 0)
        action = "right"
        next_state = (0, 1)
        reward = 1
        done = False

        # Old Q-value
        old_q_value = self.agent.q_table[state][action]

        # Update Q-value
        self.agent.update_q_value(state, action, reward, next_state, done)

        # Expected Q-value update (Bellman equation)
        best_next_action = max(self.agent.q_table[next_state], key=self.agent.q_table[next_state].get)
        expected_q_value = old_q_value + self.agent.alpha * (
            reward + self.agent.gamma * self.agent.q_table[next_state][best_next_action] - old_q_value
        )

        # Assert the updated Q-value is correct
        self.assertAlmostEqual(self.agent.q_table[state][action], expected_q_value, places=5)

    def test_q_learning_convergence(self):
        """Train the agent and check if it improves over episodes."""
        for _ in range(1000):  # Run a simple training loop
            state = self.env.reset()
            done = False
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.agent.update_q_value(state, action, reward, next_state, done)
                state = next_state

        # Check if Q-values have nonzero updates (agent has learned)
        nonzero_q_values = sum(
            1 for state in self.agent.q_table for action in self.agent.q_table[state] if self.agent.q_table[state][action] != 0
        )
        self.assertGreater(nonzero_q_values, 10)  # Should have learned some values

if __name__ == "__main__":
    unittest.main()
