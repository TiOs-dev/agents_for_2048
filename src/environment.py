import gymnasium as gym
import numpy as np

import game_2048 as game


class Game2048Env(gym.Env):
    def __init__(self, max_turns=1000):
        self._board = game.Game2048()
        self._board.initialize(False)

        self._remaining_turns = max_turns

        self.observation_space = gym.spaces.Box(
            low=0,
            high=65536,
            shape=(4, 4),
            dtype=np.uint32
        )

        self.action_space = gym.spaces.Discrete(4)

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

        return observation, reward, terminated, truncated, info
