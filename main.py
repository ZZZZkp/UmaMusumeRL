import random

import numpy as np
from typing import Any, List, Sequence, Tuple
import collections
import tqdm
import statistics

import tensorflow as tf

from game.game import Game

from neural_network.actor_critic import ActorCritic

TURN = []
game = Game()

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()


def main():
    # TODO: 开始游戏
    num_actions = 6
    num_hidden_units = 128

    model = ActorCritic(num_actions, num_hidden_units)
    # 读取之前保存的模型
    # model = tf.keras.models.load_model('UmaTrainModel.tf')

    target_model = ActorCritic(num_actions, num_hidden_units)

    min_episodes_criterion = 100
    max_episodes = 4000
    max_steps_per_episode = 78

    # consecutive trials
    reward_threshold = 12000
    running_reward = 0

    # Discount factor for future rewards
    gamma = 1  # 未来的属性和现在的属性一样，不衰减

    # Keep last episodes reward
    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

    initial_learning_rate = 0.0002
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.98,
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    with tqdm.trange(max_episodes) as t:
        for i in t:
            # if i % 10 == 0:
            #     target_model.set_weights(uma_model.get_weights())

            initial_state = tf.constant(game.reset(), dtype=tf.float32)
            episode_reward = int(train_step(
                initial_state, model, target_model, optimizer, gamma, max_steps_per_episode, i))

            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)

            t.set_description(f'Episode {i}')
            t.set_postfix(
                episode_reward=episode_reward, running_reward=running_reward)

            # Show average episode reward every 10 episodes
            if (i+1) % 200 == 0:
                print('保存训练结果')
                model.save('UmaTrainModel.tf', save_format="tf")

            if running_reward > reward_threshold and i >= min_episodes_criterion:
                break

    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')


def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns state, reward and done flag given an action."""

    state, reward, done = game.step(action)
    return (state.astype(np.float32),
            np.array(reward, np.int32),
            np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action],
                             [tf.float32, tf.int32, tf.int32])


def run_episode(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        target_model: tf.keras.Model,
        max_steps: int,
        episode_count: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    actions = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)

        # Run the uma_model and to get action probabilities and critic value
        action_logits_t, value = model(state)
        # action_logits_t, value = target_model(state)
        # tf.print(action_logits_t)
        # Sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[0, 0]

        action_probs_t = tf.nn.softmax(action_logits_t)

        # Store critic values
        values = values.write(t, tf.squeeze(value))

        # Store log probability of the action chosen
        action_probs = action_probs.write(t, action_probs_t[0, action])
        actions = actions.write(t, action)

        # Apply action to the environment to get next state and reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs.stack()
    tf.print(action_probs)
    values = values.stack()
    rewards = rewards.stack()

    actions = actions.stack()
    tf.print(actions)

    return action_probs, values, rewards


def get_expected_return(
        rewards: tf.Tensor,
        gamma: float,
        standardize: bool = True) -> tf.Tensor:
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) /
                   (tf.math.reduce_std(returns) + eps))

    return returns


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


def compute_loss(
        action_probs: tf.Tensor,
        values: tf.Tensor,
        returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined actor-critic loss."""

    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss


@tf.function
def train_step(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        target_model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        gamma: float,
        max_steps_per_episode: int,
        episode_count: int) -> tf.Tensor:
    """Runs a uma_model training step."""

    with tf.GradientTape() as tape:
        # Run the uma_model for one episode to collect training data
        action_probs, values, rewards = run_episode(
            initial_state, model, target_model, max_steps_per_episode, episode_count)

        # Calculate expected returns
        returns = get_expected_return(rewards, gamma)

        # Convert training data to appropriate TF tensor shapes
        action_probs, values, returns = [
            tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

        # Calculating loss values to update our network
        loss = compute_loss(action_probs, values, returns)

    # Compute the gradients from the loss
    grads = tape.gradient(loss, model.trainable_variables)

    # Apply the gradients to the uma_model's parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward


if __name__ == '__main__':
    main()
