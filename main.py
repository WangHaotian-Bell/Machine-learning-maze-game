import argparse
import time
import pygame
from environment import MazeEnv
from agent import QLearningAgent
from gui import MazeGame


def train_agent(env, agent, episodes=500):

    print("Training agent...")
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if episode % 50 == 0:
            print(f"Episode {episode}/{episodes}, Total Reward: {total_reward:.2f}")

    agent.save_q_table()
    print("Training complete. Q-table saved.")


def run_game(grid_size=(10, 10), start=(0, 0), goal=(9, 9), obstacles=None, train=False, episodes=500):

    env = MazeEnv(grid_size, start, goal, obstacles)
    agent = QLearningAgent(env)

    if train:
        agent.load_q_table()  # Load Q-table if it exists
        train_agent(env, agent, episodes)
        agent.save_q_table()  # Save after training

    # Initialize Pygame visualization
    game = MazeGame(grid_size, start, goal, obstacles)
    game.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-learning Maze Game")
    parser.add_argument("--train", action="store_true", help="Train the agent before running")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    args = parser.parse_args()

    # Example maze setup
    grid_size = (10, 10)
    start = (0, 0)
    goal = (9, 9)
    obstacles = [(1, 3),(4, 7),(2, 5),(7, 1),(3, 9),(6, 2),(8, 4),(9, 6),(5, 8),(3, 1),(7, 7),(2, 8),(1, 4),(6, 3),(4, 5),(9, 2),(8, 7),(5, 1),(3, 6),(7, 9),(6, 9)]

    # Run the game with or without training
    run_game(grid_size, start, goal, obstacles, train=args.train, episodes=args.episodes)
