"""
This module implements an interface for the environments used in this training framework.
"""

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym


class EnvironmentBase(ABC, gym.Env):
    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
