from typing import Any
from pathlib import Path
from os.path import join
import tomllib

import numpy as np
import gymnasium as gym
import pygame
import game_2048 as game

from rl_training_framework.environments.environment_base import EnvironmentBase


class Env2048(EnvironmentBase):
    def __init__(
        self,
        game_config: str | dict[str, Any] = join(
            Path(__file__).resolve().parent, "game_config.toml"
        ),
        render_mode: str = "human",
        max_turns=1000,
    ):
        if isinstance(game_config, str):
            with open(game_config, "rb") as config_file:
                self._game_config = tomllib.load(config_file)
        else:
            self._game_config = game_config

        self.reset()

        self._remaining_turns = max_turns

        self._observation_space = gym.spaces.Box(
            low=self._game_config["observation_space"]["low"],
            high=self._game_config["observation_space"]["high"],
            shape=self._game_config["observation_space"]["shape"],
            dtype=np.uint32,
        )

        self._action_space = gym.spaces.Discrete(4)
        self._render_mode = render_mode

        if self._render_mode == "human":
            self._screen_size = (
                self._game_config["rendering"]["number_of_tiles"]
                * self._game_config["rendering"]["tile_size"]
                + (self._game_config["rendering"]["number_of_tiles"] + 1)
                * self._game_config["rendering"]["gap_between_tiles"]
                + 2 * self._game_config["rendering"]["margin_size"]
            )

            pygame.init()
            pygame.display.init()
            self._font = pygame.font.SysFont(
                self._game_config["rendering"]["font"],
                self._game_config["rendering"]["font_size"],
            )

            self._screen = pygame.Surface((self._screen_size, self._screen_size))

            self._window = pygame.display.set_mode(
                (self._screen_size, self._screen_size)
            )

            self.clock = pygame.time.Clock()

    def _get_obs(self):
        return np.array(self._board.board, dtype=np.uint32)

    def _get_info(self):
        return self._board.score, self._board.num_turns

    def reset(self):
        self._board = game.Game2048()
        self._board.initialize(False)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: int):
        if action == 0:
            did_turn = self._board.up()
        elif action == 1:
            did_turn = self._board.right()
        elif action == 2:
            did_turn = self._board.down()
        else:
            did_turn = self._board.left()

        if did_turn:
            self._board.generate_new_tile()

        observation = self._get_obs()
        info = self._get_info()

        self._remaining_turns -= 1
        truncated = self._remaining_turns == 0
        terminated = not did_turn

        reward = np.max(observation) if truncated or terminated else 0

        if self._render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        self._draw_board()
        self._window.blit(self._screen, self._screen.get_rect())
        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(self.metadata["render_fps"])

    def _draw_tile(self, value: int, x: int, y: int):
        color = self._game_config["rendering"]["tile_colors"].get(
            str(value), (60, 58, 50)
        )
        rect = pygame.Rect(
            x,
            y,
            self._game_config["rendering"]["tile_size"],
            self._game_config["rendering"]["tile_size"],
        )
        pygame.draw.rect(self._screen, color, rect)
        if value != 0:
            text = self._font.render(
                str(value), True, self._game_config["rendering"]["font_color"]
            )
            text_rect = text.get_rect(
                center=(
                    x + self._game_config["rendering"]["tile_size"] / 2,
                    y + self._game_config["rendering"]["tile_size"] / 2,
                )
            )
            self._screen.blit(text, text_rect)

    def _draw_board(self):
        self._screen.fill(self._game_config["rendering"]["background_color"])
        for row in range(self._game_config["rendering"]["number_of_tiles"]):
            for col in range(self._game_config["rendering"]["number_of_tiles"]):
                value = self._board.board[row][col]
                x = (
                    self._game_config["rendering"]["margin_size"]
                    + self._game_config["rendering"]["gap_between_tiles"]
                    + col
                    * (
                        self._game_config["rendering"]["tile_size"]
                        + self._game_config["rendering"]["gap_between_tiles"]
                    )
                )
                y = (
                    self._game_config["rendering"]["margin_size"]
                    + self._game_config["rendering"]["gap_between_tiles"]
                    + row
                    * (
                        self._game_config["rendering"]["tile_size"]
                        + self._game_config["rendering"]["gap_between_tiles"]
                    )
                )
                self._draw_tile(value, x, y)

    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    from random import randrange

    env = Env2048()
    for i in range(10000):
        action = randrange(0, 4)
        env.step(action)
