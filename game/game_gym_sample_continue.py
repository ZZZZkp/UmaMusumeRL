from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from torch.distributions import Categorical

from game.game_gym import Game


class SampleContinueGame(Game):
    def __init__(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().__init__(seed=seed, options=options)

    def step(self, action):
        action = torch.tensor(action)

        action_softmax = action.softmax(-1)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(action_softmax)

        # and sample an action using the distribution
        action_code = m.sample()
        status_to_net, reward, terminated, truncated, info = super().step(action_code)

        return status_to_net, reward, terminated, truncated, info

    def set_space(self, net):
        self.action_space = spaces.Box(-10, 10, shape=(7,), dtype=np.float64)
        self.observation_space = spaces.Box(0, 1, shape=(net.shape[0],), dtype=np.float64)
