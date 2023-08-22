import tensorflow as tf
from typing import Any, List, Sequence, Tuple

from tensorflow.python.keras import models, layers, optimizers


class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(
            self,
            num_actions: int,
            num_hidden_units: int):
        """Initialize."""
        super().__init__()
        self.dense1 = layers.Dense(num_hidden_units, activation="relu")
        self.dense2 = tf.keras.layers.BatchNormalization()
        self.dense3 = layers.Dense(num_hidden_units, activation="relu")
        self.dense4 = tf.keras.layers.BatchNormalization()
        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor,  training=None, mask=None) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.actor(x), self.critic(x)
