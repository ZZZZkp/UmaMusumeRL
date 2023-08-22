# Copy from https://github.com/vmorarji/TD3-TensorFlow/
import math

import gym
import os

import numpy as np

from game.game import Game
from model import TD3
from replay_buffer import ReplayBuffer
import datetime as dt
import tensorflow as tf

current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.exists('./logs/' + current_time):
    os.makedirs('./logs/' + current_time)

if not os.path.exists('./models/' + current_time):
    os.makedirs('./models/' + current_time)

# initialise the environment
env = Game()
action_dim = 6
state_dim = env.reset().shape[0]
result_writer = tf.summary.create_file_writer('./logs/' + current_time)


def evaluate_policy(policy, eval_episodes=10):
    # during training the policy will be evaluated without noise
    avg_reward = 0.
    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        while not done:
            action_logits = policy.select_action(state)
            action = tf.random.categorical(action_logits, 1)[0, 0]
            state, reward, done = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("---------------------------------------")
    print("Average Reward over the Evaluation Step: %f" % (avg_reward))
    print("---------------------------------------")
    return avg_reward


# initialise the replay buffer
memory = ReplayBuffer(2000)
# initialise the policy
policy = TD3(state_dim, action_dim, current_time=current_time, summaries=True)

# 读取保存的模型进度, 第一个参数为文件夹名，第二个参数为模型最后一位的数字
# policy.load('20230822-000647', 9)

max_steps = 2e6
start_steps = 1e4
total_steps = 0
eval_freq = 5e3
save_freq = 1e5
eval_counter = 0
episode_num = 0
episode_reward = 0
done = True

while total_steps < max_steps:

    if done:

        # print the results at the end of the episode
        if total_steps != 0:
            print('Episode: {}, Total Timesteps: {}, Episode Reward: {:.2f}'.format(
                episode_num,
                total_steps,
                episode_reward
            ))
            with result_writer.as_default():
                tf.summary.scalar('total_reward', episode_reward, step=episode_num)

        if eval_counter > eval_freq:
            eval_counter %= eval_freq
            evaluate_policy(policy)

        state = env.reset()

        done = False
        episode_reward = 0
        episode_steps = 0
        episode_num += 1

    # the environment will play the initial episodes randomly
    # if total_steps < start_steps:
    #     action_logits = tf.math.log(tf.random.uniform([6,1], minval=0, maxval= math.e))
    # else:  # select an action from the actor network with noise

    action_logits = policy.select_action(state)

    action = tf.random.categorical(action_logits, 1)[0, 0]
    action_logits = tf.squeeze(action_logits)
    # the agent plays the action
    next_state, reward, done= env.step(action)

    # add to the total episode reward
    episode_reward += reward

    # check if the episode is done
    done_bool = 0 if episode_steps + 1 == env.max_steps else float(done)

    # add to the memory buffer
    memory.add((state, next_state, action_logits, reward, done_bool))

    # update the state, episode timestep and total timestep
    state = next_state
    episode_steps += 1
    total_steps += 1
    eval_counter += 1

    # train after the first episode
    if total_steps > start_steps:
        policy.train(memory, batch_size=512)

    # save the uma_model
    if total_steps % save_freq == 0:
        policy.save(int(total_steps / save_freq))