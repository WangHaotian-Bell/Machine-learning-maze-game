from environment import MazeEnv
from agent import QLearningAgent


def train_agent(env, agent, episodes=1000):
    agent.load_q_table()  # Load existing Q-table (if available)

    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state, done)
            state = next_state

        if episode % 50 == 0:
            print(f"Episode {episode}/{episodes} completed.")

    agent.save_q_table()
    print("Training complete. Q-table saved.")


if __name__ == "__main__":
    env = MazeEnv(grid_size=(10, 10), start=(0, 0), goal=(9, 9), obstacles=[(1, 3),(4, 7),(2, 5),(7, 1),(3, 9),(6, 2),(8, 4),(9, 6),(5, 8),(3, 1),(7, 7),(2, 8),(1, 4),(6, 3),(4, 5),(9, 2),(8, 7),(5, 1),(3, 6),(7, 9),(6, 9)])
    agent = QLearningAgent(env)
    train_agent(env, agent, episodes=1000)
