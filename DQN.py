# Copy from https://geektutu.com/post/tensorflow2-gym-dqn.html

from collections import deque
import random
import gym
import numpy as np
from tensorflow.python.keras import models, layers, optimizers
import tensorflow as tf
from typing import Any, List, Sequence, Tuple

from game.game import Game


class DQN(object):
    def __init__(self):
        self.step = 0
        self.update_freq = 200  # 模型更新频率
        self.replay_size = 200  # 训练集大小
        self.replay_queue = deque(maxlen=self.replay_size)
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        """创建一个隐藏层为100的神经网络"""
        STATE_DIM, ACTION_DIM = 50, 6
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, input_dim=STATE_DIM, activation='elu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='elu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='elu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='elu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(ACTION_DIM, activation="linear")
        ])
        model.compile(loss='mean_squared_error',
                      optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    def act(self, s, epsilon=0.1):
        """预测动作"""
        # 刚开始时，加一点随机成分，产生更多的状态
        if np.random.uniform() < epsilon - self.step * 0.0002:
            return np.random.choice([0, 1, 2])
        return np.argmax(self.model.predict(np.array([s]), verbose=0)[0])

    def save_model(self, file_path='UmaTrainModel-v0-dqn.h5'):
        print('uma_model saved')
        self.model.save(file_path)

    def remember(self, s, a, next_s, reward):
        """历史记录，position >= 0.4时给额外的reward，快速收敛"""
        if next_s[0] >= 0.4:
            reward += 1
        self.replay_queue.append((s, a, next_s, reward))

    def train(self, batch_size=64, lr=1, factor=0.95):
        if len(self.replay_queue) < self.replay_size:
            return
        self.step += 1
        # 每 update_freq 步，将 uma_model 的权重赋值给 target_model
        if self.step % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        replay_batch = random.sample(self.replay_queue, batch_size)
        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])

        Q = self.model.predict(s_batch, verbose=0)
        Q_next = self.target_model.predict(next_s_batch, verbose=0)

        # 使用公式更新训练集中的Q值
        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + factor * np.amax(Q_next[i]))

        # 传入网络进行训练
        self.model.fit(s_batch, Q, verbose=0, callbacks=[Callback()])


class Callback(tf.keras.callbacks.Callback):
    SHOW_NUMBER = 10
    counter = 0
    epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        if self.counter == self.SHOW_NUMBER or self.epoch == 1:
            print('Epoch: ' + str(self.epoch) + ' loss: ' + str(logs['loss']))
            if self.epoch > 1:
                self.counter = 0
        self.counter += 1


env = Game()
episodes = 1000  # 训练1000次
score_list = []  # 记录所有分数
agent = DQN()
for i in range(episodes):
    s = env.reset()
    print(s.shape)
    score = 0
    while True:
        a = agent.act(s)
        next_s, reward, done = env.step(a)
        agent.remember(s, a, next_s, reward)
        agent.train()
        score += reward
        s = next_s
        if done:
            score_list.append(score)
            print('episode:', i, 'score:', score, 'max:', max(score_list))
            break
    # 最后10次的平均分大于 12000 时，停止并保存模型
    if np.mean(score_list[-10:]) > 12000:
        agent.save_model()
        break
