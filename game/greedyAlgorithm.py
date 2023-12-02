import numpy as np

from action.goTraining import calculate_train_effect_v2
from game import Game
from uma_model.status import Status

game = Game()
game.reset()


def selectAction(current_state: Status):
    max_reward = 0
    action = 0
    if current_state.energy < 50:
        return 5
    else:
        for i in np.arange(5):
            train_effect = calculate_train_effect_v2(i, current_state)
            reward = game.calculate_reward(True, train_effect)
            if (max_reward < reward):
                max_reward = reward
                action = i
    return action

mean_total_reward = 0
for i in range(100):
    game.reset()
    done = False
    total_reward = 0
    game.character_status.add_motivation(2)
    while not done:
        _, reward, done = game.step(selectAction(game.character_status))
        total_reward += reward
    mean_total_reward += total_reward
    print('total_reward: {}'.format(total_reward))
mean_total_reward = mean_total_reward/100
print('mean_total_reward: {}'.format(mean_total_reward))
