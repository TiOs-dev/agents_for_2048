import random
import math
from collections import namedtuple, deque

import torch.nn.functional as F
import torch.nn as nn
import torch

TRANSITION = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity):
        self._memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a Transition"""
        self._memory.append(TRANSITION(*args))

    def sample(self, batch_size):
        return random.sample(self._memory, batch_size)

    def __len__(self):
        return len(self._memory)


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
