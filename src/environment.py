import gymnasium as gym
import numpy as np
import pygame

import game_2048 as game


class Game2048Env(gym.Env):
    metadata = {"render_modes": ["human", "sth_else"], "render_fps": 4}

    def __init__(self, max_turns=1000, render_mode="human"):
        self._board = game.Game2048()
        self._board.initialize(False)

        self._remaining_turns = max_turns

        self.observation_space = gym.spaces.Box(
            low=0, high=65536, shape=(4, 4), dtype=np.uint32
        )

        self.action_space = gym.spaces.Discrete(4)

        # pygame stuff
        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
            self.size = 4
            self.tile_size = 100
            self.gap_size = 10
            self.margin = 20
            self.screen_size = (
                self.size * self.tile_size
                + (self.size + 1) * self.gap_size
                + 2 * self.margin
            )
            self.screen_width = self.screen_size
            self.screen_height = self.screen_size
            self.background_color = (255, 251, 240)
            self.empty_tile_color = (205, 192, 180)
            self.tile_colors = {
                2: (238, 228, 218),
                4: (237, 224, 200),
                8: (242, 177, 121),
                16: (245, 149, 99),
                32: (246, 124, 95),
                64: (246, 94, 59),
                128: (237, 207, 114),
                256: (237, 204, 97),
                512: (237, 200, 80),
                1024: (237, 197, 63),
                2048: (237, 194, 46),
            }
            self.font_color = (0, 0, 0)
            self.font = pygame.font.SysFont("arial", 40)

            self.screen = pygame.Surface((self.screen_width, self.screen_height))

            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
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

    def step(self, action):
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

        reward = np.max(observation) if did_turn else 0

        truncated = False

        self._remaining_turns -= 1
        terminated = self._remaining_turns == 0

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        self._draw_board()
        self.window.blit(self.screen, self.screen.get_rect())
        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(self.metadata["render_fps"])

    def _draw_tile(self, value, x, y):
        color = self.tile_colors.get(value, (60, 58, 50))
        rect = pygame.Rect(x, y, self.tile_size, self.tile_size)
        pygame.draw.rect(self.screen, color, rect)
        if value != 0:
            text = self.font.render(str(value), True, self.font_color)
            text_rect = text.get_rect(
                center=(x + self.tile_size / 2, y + self.tile_size / 2)
            )
            self.screen.blit(text, text_rect)

    def _draw_board(self):
        self.screen.fill(self.background_color)
        for row in range(self.size):
            for col in range(self.size):
                value = self._board.board[row][col]
                x = self.margin + self.gap_size + col * (self.tile_size + self.gap_size)
                y = self.margin + self.gap_size + row * (self.tile_size + self.gap_size)
                self._draw_tile(value, x, y)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
