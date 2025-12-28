from typing import Any
from pathlib import Path
from os.path import join
import tomllib
import random
import math

import torch
import torch.optim as optim
import torch.nn as nn

from rl_training_framework.environments.environment_base import EnvironmentBase
from rl_training_framework.agents.agent_base import AgentBase
from rl_training_framework.agents.AgentDeepQLearning.utils import (
    DQN,
    ReplayMemory,
    TRANSITION,
)


class Agent(AgentBase):
    def __init__(
        self,
        env: EnvironmentBase,
        hyperparams: str | dict[str, Any] = join(
            Path(__file__).resolve().parent, "hyperparameters.toml"
        ),
    ):
        super().__init__(env, hyperparams)
        if isinstance(hyperparams, str):
            with open(hyperparams, "rb") as hyperp_file:
                self.hyperparams = tomllib.load(hyperp_file)
        else:
            self.hyperparams = hyperparams
        self.steps_done = 0
        self.device = "cpu"

        self.setup_training()

    def select_action(self, state: torch.Tensor):
        sample = random.random()
        eps_threshold = self.hyperparams["eps_end"] + (
            self.hyperparams["eps_start"] - self.hyperparams["eps_end"]
        ) * math.exp(-1.0 * self.steps_done / self.hyperparams["eps_decay"])
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                result = self.policy_net(state).max(1).indices.view(1, 1)
                return result
        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.hyperparams["batch_size"]:
            return
        transitions = self.memory.sample(self.hyperparams["batch_size"])
        batch = TRANSITION(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)

        reward_batch = torch.cat([reward.to(torch.uint32) for reward in batch.reward])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(
            self.hyperparams["batch_size"], device=self.device
        )
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        expected_state_action_values = (
            next_state_values * self.hyperparams["gamma"]
        ) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def setup_training(self):
        n_actions = self.env.action_space.n
        state, info = self.env.reset()
        n_observations = len(torch.flatten(torch.tensor(state)))
        self.policy_net = DQN(n_observations, n_actions)
        self.target_net = DQN(n_observations, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.hyperparams["lr"], amsgrad=True
        )

        self.memory = ReplayMemory(10000)

        self.rewards = []
        self.state = torch.tensor(state, dtype=torch.float32, device="cpu").unsqueeze(0)

    def run_trajectory(self):
        state, info = self.env.reset()
        self.state = torch.tensor(state, dtype=torch.float32, device="cpu").unsqueeze(0)

        for t in tqdm(
            range(1000), desc="curr run", position=1, leave=False
        ):  # TODO: How do I deal with truncation (range(1000))? Do I control it here or in the environment?
            action = test_agent.select_action(self.state)
            observation, reward, terminated, truncated, _ = self._env.step(
                action.item()
            )
            self.rewards.append(reward)
            reward = torch.tensor([reward], device="cpu")
            done = terminated or truncated

            if terminated:
                next_state = None
                break

            next_state = torch.tensor(
                observation, dtype=torch.float32, device="cpu"
            ).unsqueeze(0)

            self.memory.push(self.state, action, next_state, reward)

            self.state = next_state

            self.optimize_model()

            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * self.hyperparams["tau"] + target_net_state_dict[key] * (
                    1 - self.hyperparams["tau"]
                )
            self.target_net.load_state_dict(target_net_state_dict)

            if done:
                next_state = None
                break

    def do_test_run(self):
        pass


if __name__ == "__main__":
    from tqdm import tqdm

    from rl_training_framework.environments.Env2048.environment import Env2048

    num_episodes = 50
    test_env = Env2048()
    test_agent = Agent(test_env)

    rewards = []

    for i_episode in tqdm(
        range(num_episodes), desc="episodes", position=0, leave=False
    ):
        test_agent.run_trajectory()

    print("Complete!")
