import logging
from random import randint
from enum import Enum

import numpy as np
import gymnasium as gym
import pygame

from rl_training_framework.environments.environment_base import EnvironmentBase

logger = logging.getLogger("TrainingLogger")


class Move(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class BlockWorld(EnvironmentBase):
    def __init__(self, render_mode="human"):
        self.width = 10
        self.height = 10
        self.reset()

        self.render_mode = render_mode
        if self.render_mode == "human":
            self.tile_size = 100
            self.gap_size = 10
            self.margin_size = 20

            self.screen_width = (
                self.width * self.tile_size
                + (self.width + 1) * self.gap_size
                + 2 * self.margin_size
            )

            self.screen_height = (
                self.height * self.tile_size
                + (self.height + 1) * self.gap_size
                + 2 * self.margin_size
            )

            pygame.init()
            pygame.display.init()

            self._screen = pygame.Surface((self.screen_width, self.screen_height))

            self._window = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )

            self.clock = pygame.time.Clock()

    def reset(self):
        self.start = (randint(0, 9), randint(0, 9))
        self.stop = (randint(0, 9), randint(0, 9))
        self.path = [
            self.start,
        ]  #
        self.num_of_turns = 0

    def step(self, action: int):
        if action == Move.UP.value:
            self.path.append((self.path[-1][0], self.path[-1][1] - 1))
        elif action == Move.RIGHT.value:
            self.path.append((self.path[-1][0] + 1, self.path[-1][1]))
        elif action == Move.DOWN.value:
            self.path.append((self.path[-1][0], self.path[-1][1] + 1))
        elif action == Move.LEFT.value:
            self.path.append((self.path[-1][0] - 1, self.path[-1][1]))
        else:
            raise ValueError(
                f"You chose action-value {action}. \
                This is not a valid action. \
                Please chose from the following: \
                up (0), right (1), down (2), left (3)"
            )

        if self.path[-1] == self.stop:
            reward = 10
            terminated = True
            self.reset()
        elif (
            self.path[-1][0] >= self.width
            or self.path[-1][0] < 0
            or self.path[-1][1] >= self.height
            or self.path[-1][1] < 0
        ):
            reward = -10
            terminated = True
            self.reset()
        else:
            reward = -1
            terminated = False

        self.num_of_turns += 1

        if self.render_mode == "human":
            self.render()

        return self.path[:], reward, terminated, None, self.num_of_turns

    def render(self):
        self._draw_board()
        self._window.blit(self._screen, self._screen.get_rect())
        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(4.0)  # number of FPS

    def _convert_to_screen_coord(self, value):
        return (
            self.margin_size + self.gap_size + value * (self.tile_size + self.gap_size)
        )

    def _draw_tile(self, x, y):
        color_start = (0, 100, 0)
        color_stop = (100, 0, 0)
        color_path = (0, 0, 100)
        color_empty = (205, 192, 180)

        rect = pygame.Rect(
            self._convert_to_screen_coord(x),
            self._convert_to_screen_coord(y),
            self.tile_size,
            self.tile_size,
        )

        if (x, y) == self.start:
            pygame.draw.rect(self._screen, color_start, rect)
            print("Start")
        elif (x, y) == self.stop:
            pygame.draw.rect(self._screen, color_stop, rect)
            print("Stop")
        elif (x, y) in self.path:
            pygame.draw.rect(self._screen, color_path, rect)
            print("Path")
        else:
            pygame.draw.rect(self._screen, color_empty, rect)

    def _draw_board(self):
        self._screen.fill((255, 251, 240))
        for x in range(self.width):
            for y in range(self.height):
                self._draw_tile(x, y)

    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()


def main():
    """
    Function for testing the environment.
    """
    env = BlockWorld()
    for i in range(10):
        action = randint(0, 3)
        env.step(action)


if __name__ == "__main__":
    main()
