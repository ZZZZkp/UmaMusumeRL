# Copy from https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
import argparse
import collections
import datetime
import os
import statistics

import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from game.game import Game

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
# 未来回报衰减系数
parser.add_argument('--gamma', type=float, default=1.0, metavar='G',
                    help='discount factor (default: 1.0)')
# 保存路径
parser.add_argument('--save-path', type=str, default='actor_critic_model', metavar='S',
                    help='model save path')
# 加载路径（和保存路径不一样，要加上文件名）
parser.add_argument('--load-path', type=str, default=None, metavar='S',
                    help='model load path')
parser.add_argument('--episode', type=int, default=2000, metavar='E',
                    help='max episode')
# 打印记录间隔
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='interval between training status logs (default: 100)')
# 保存模型间隔
parser.add_argument('--save-interval', type=int, default=1000, metavar='N',
                    help='interval between saving model (default: 1000)')
args = parser.parse_args()


env = Game()
env.reset()
state_size = env.reset().shape[0]
action_size = 7

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(state_size, 512)
        self.affine2 = nn.Linear(512, 256)
        self.affine3 = nn.Linear(256, 128)

        # actor's layer
        self.action_head = nn.Linear(128, action_size)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


model = Policy()
if args.load_path != None:
    model = torch.load(args.load_path)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    episodes_reward: collections.deque = collections.deque(maxlen=100)

    # run infinitely many episodes
    for i_episode in range(1, args.episode):

        # reset environment and episode reward
        state = env.reset(log_every_game=False)
        ep_reward = 0

        # for each episode, only run 78 steps so that we don't
        # infinite loop while learning
        for t in range(1, 78):

            # select action from policy
            action = select_action(state)

            # take the action
            state, reward, done = env.step(action)

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        episodes_reward.append(ep_reward)
        running_reward = statistics.mean(episodes_reward)
        # perform backprop
        finish_episode()

        # log results
        if i_episode % args.log_interval == 0:
            print('上次的最终属性')
            env.print_current_status()
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        if i_episode % args.save_interval == 0:
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)
            torch.save(model, os.path.join(args.save_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pkl'))

        if running_reward > 12000:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
