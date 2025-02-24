import pygame
import os
import time
import random
from environment import MazeEnv
from agent import QLearningAgent

TILE_SIZE = 50
FPS = 5


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (192, 192, 192)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class MazeGame:
    def __init__(self, grid_size=(10, 10), start=(0, 0), goal=(9, 9), obstacles=None):
        pygame.init()
        self.env = MazeEnv(grid_size, start, goal, obstacles)
        self.agent = QLearningAgent(self.env)
        self.screen_size = (grid_size[1] * TILE_SIZE, grid_size[0] * TILE_SIZE)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Q-Learning Maze Game")
        self.clock = pygame.time.Clock()
        self.training_mode = True

        if os.path.exists("models/q_table.pkl"):
            self.agent.load_q_table()
        else:
            print("Warning: No trained Q-table found. Running without training.")

    def draw_grid(self):
        for x in range(self.env.grid_size[0]):
            for y in range(self.env.grid_size[1]):
                rect = pygame.Rect(y * TILE_SIZE, x * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                if (x, y) in self.env.obstacles:
                    pygame.draw.rect(self.screen, RED, rect)
                elif (x, y) == self.env.goal:
                    pygame.draw.rect(self.screen, GREEN, rect)
                else:
                    pygame.draw.rect(self.screen, WHITE, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 1)

        ax, ay = self.env.state
        pygame.draw.circle(self.screen, BLUE, (ay * TILE_SIZE + TILE_SIZE // 2, ax * TILE_SIZE + TILE_SIZE // 2), TILE_SIZE // 3)

    def draw_parameters(self):
        font = pygame.font.Font(None, 24)
        parameters = [
            f"Alpha (Learning Rate): {self.agent.alpha}",
            f"Gamma (Discount Factor): {self.agent.gamma}",
            f"Epsilon (Exploration): {self.agent.epsilon}",
            f"Grid Size: {self.env.grid_size}",
            f"Agent Position: {self.env.state}",
            f"Goal Position: {self.env.goal}"
        ]

        x, y = 10, self.screen_size[1] - (len(parameters) * 20) - 10

        for param in parameters:
            text_surface = font.render(param, True, BLACK)
            self.screen.blit(text_surface, (x, y))
            y += 20

    def run(self):
        running = True
        self.env.reset()

        while running:
            self.screen.fill(GRAY)
            self.draw_grid()
            self.draw_parameters()
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

            if self.training_mode:
                action = self.agent.choose_action(self.env.state)
            else:
                action = self.get_manual_input()

            next_state, _, done = self.env.step(action)
            if done:
                time.sleep(1)
                self.env.reset()

        pygame.quit()

    def get_manual_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            return 'up'
        if keys[pygame.K_DOWN]:
            return 'down'
        if keys[pygame.K_LEFT]:
            return 'left'
        if keys[pygame.K_RIGHT]:
            return 'right'
        return random.choice(self.env.get_action_space())

if __name__ == "__main__":
    game = MazeGame(grid_size=(10, 10), start=(0, 0), goal=(9, 9), obstacles=[(1, 3),(4, 7),(2, 5),(7, 1),(3, 9),(6, 2),(8, 4),(9, 6),(5, 8),(3, 1),(7, 7),(2, 8),(1, 4),(6, 3),(4, 5),(9, 2),(8, 7),(5, 1),(3, 6),(7, 9),(6, 9)])
    game.run()
