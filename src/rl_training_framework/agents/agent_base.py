from abc import ABC, abstractmethod
from typing import Any
import tomllib

from rl_training_framework.environments.environment_base import EnvironmentBase


class AgentBase(ABC):
    def __init__(self, environment: EnvironmentBase, hyperparams: str | dict[str, Any]):
        self._env = environment

        if isinstance(hyperparams, str):
            with open(hyperparams, "rb") as hyperparams_file:
                self._hyperparams = tomllib.load(hyperparams_file)
        else:
            self._hyperparams = hyperparams

    @property
    def env(self):
        return self._env

    @env.setter
    def environment(self, new_env: EnvironmentBase):
        self._env = new_env

    @abstractmethod
    def select_action(self):
        pass

    @abstractmethod
    def optimize_model(self):
        pass

    @abstractmethod
    def setup_training(self):
        pass

    @abstractmethod
    def run_trajectory(self):
        pass

    @abstractmethod
    def do_test_run(self):
        pass
