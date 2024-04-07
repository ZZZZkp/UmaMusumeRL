from typing import Any

import numpy as np
import random

from gymnasium import spaces
from gymnasium.core import ObsType


from action.goOut import go_out
from action.goTraining import go_training, update_facilities, add_cards_bond_v2, add_stats
from action.rest import rest
from game.game_gym import Game
from uma_model.character import Character
from uma_model.compositionInformation import CompositionInformation
from uma_model.status import Status
from neural_network import neural_network
import gymnasium as gym


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
