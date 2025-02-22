import pygame
import os
import time
import random
from environment import MazeEnv
from agent import QLearningAgent

# Pygame configuration
TILE_SIZE = 50  # Pixel size of each grid cell
FPS = 5  # Frames per second

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (192, 192, 192)
BLUE = (0, 0, 255)  # Agent color
GREEN = (0, 255, 0)  # Goal color
RED = (255, 0, 0)  # Wall color

class MazeGame:
    def __init__(self, grid_size=(10, 10), start=(0, 0), goal=(9, 9), obstacles=None):
        """Initialize the maze game with the environment and agent."""
        pygame.init()
        self.env = MazeEnv(grid_size, start, goal, obstacles)
        self.agent = QLearningAgent(self.env)  # ✅ Ensure the agent is properly initialized
        self.screen_size = (grid_size[1] * TILE_SIZE, grid_size[0] * TILE_SIZE)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Q-Learning Maze Game")
        self.clock = pygame.time.Clock()
        self.training_mode = True  # Toggle between training and inference mode

        # ✅ Load Q-table only if it exists
        if os.path.exists("models/q_table.pkl"):
            self.agent.load_q_table()
        else:
            print("Warning: No trained Q-table found. Running without training.")

    def draw_grid(self):
        """Draw the maze grid with obstacles, agent, and goal."""
        for x in range(self.env.grid_size[0]):
            for y in range(self.env.grid_size[1]):
                rect = pygame.Rect(y * TILE_SIZE, x * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                if (x, y) in self.env.obstacles:
                    pygame.draw.rect(self.screen, RED, rect)  # Walls
                elif (x, y) == self.env.goal:
                    pygame.draw.rect(self.screen, GREEN, rect)  # Goal
                else:
                    pygame.draw.rect(self.screen, WHITE, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 1)  # Grid lines

        # Draw agent
        ax, ay = self.env.state
        pygame.draw.circle(self.screen, BLUE, (ay * TILE_SIZE + TILE_SIZE // 2, ax * TILE_SIZE + TILE_SIZE // 2), TILE_SIZE // 3)

    def run(self):
        """Runs the Pygame visualization."""
        running = True
        self.env.reset()

        while running:
            self.screen.fill(GRAY)
            self.draw_grid()
            pygame.display.flip()
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t:
                        self.training_mode = not self.training_mode
                        print("Toggled training mode:", self.training_mode)
                    elif event.key == pygame.K_r:
                        self.env.reset()
                        print("Reset environment.")

            # Agent movement
            if self.training_mode:
                action = self.agent.choose_action(self.env.state)
            else:
                action = self.get_manual_input()

            next_state, _, done = self.env.step(action)
            if done:
                time.sleep(1)  # Pause before restarting
                self.env.reset()

        pygame.quit()

    def get_manual_input(self):
        """Allows manual movement using arrow keys."""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            return 'up'
        if keys[pygame.K_DOWN]:
            return 'down'
        if keys[pygame.K_LEFT]:
            return 'left'
        if keys[pygame.K_RIGHT]:
            return 'right'
        return random.choice(self.env.get_action_space())  # Default to random action if no key is pressed

if __name__ == "__main__":
    game = MazeGame(grid_size=(10, 10), start=(0, 0), goal=(9, 9), obstacles=[(1, 3),(4, 7),(2, 5),(7, 1),(3, 9),(6, 2),(8, 4),(9, 6),(5, 8),(3, 1),(7, 7),(2, 8),(1, 4),(6, 3),(4, 5),(9, 2),(8, 7),(5, 1),(3, 6),(7, 9),(6, 9)])
    game.run()
