from collections import namedtuple, deque
import random
import math

import torch.nn.functional as F
import torch.nn as nn
import torch


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.reshape(x, (-1, self.n_observations))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            result = policy_net(state).max(1).indices.view(1, 1)
            return result
    else:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long)


class QLearningAgent:
    def __init__(self, eps, env):
        self.eps = eps
        self.env = env

    def initialize_func_approx(self):
        self.policy_net = None

    def select_action(self, state):
        sample = random.random()
        if sample > self.eps:
            with torch.no_grad():
                result = self.policy_net(state).max(1).indices.view(1, 1)
                return result
        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)

    def optimize_model(self):
        pass
