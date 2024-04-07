from typing import Any

import numpy as np
from gymnasium import spaces

from game.game_gym import Game


class ContinueGame(Game):
    def __init__(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().__init__(seed=seed, options=options)

    def step(self, action):
        action_code = np.argmax(action)
        status_to_net, reward, terminated, truncated, info = super().step(action_code)

        return status_to_net, reward, terminated, truncated, info

    def set_space(self, net):
        self.action_space = spaces.Box(0, 1, shape=(7,), dtype=np.float64)
        self.observation_space = spaces.Box(0, 1, shape=(net.shape[0],), dtype=np.float64)
