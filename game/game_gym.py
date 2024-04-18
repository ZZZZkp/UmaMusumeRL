from typing import Any

import numpy as np
import random

from gymnasium import spaces
from gymnasium.core import ObsType


from action.goOut import go_out
from action.goTraining import go_training, update_facilities, add_cards_bond_v2, add_stats
from action.rest import rest
from uma_model.character import Character
from uma_model.compositionInformation import CompositionInformation
from uma_model.status import Status
from neural_network import neural_network
import gymnasium as gym


class Game(gym.Env):
    def __init__(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ):
        self.character_status = None
        self.total_rewards = 0
        self.log_every_game = True
        self._max_steps = 78
        self.observation_space_type = 0
        self.print_each_actions = False
        self.init_status()

        if options:
            if 'observation_space_type' in options:
                observation_space_type = options.get('observation_space_type')
                self.observation_space_type = observation_space_type
            if 'print_each_actions' in options:
                self.print_each_actions = options.get('print_each_actions')
            if 'log_every_game' in options:
                    log_every_game = options.get('log_every_game')
                    self.log_every_game = log_every_game

        self.reset(seed=seed, options=options)

    def random_distribution_support_cards(self):
        cards = self.character_status.composition_information.support_card_list
        support_cards_distribution = np.zeros((6, 6), dtype=int)  # 最外层表示支援卡分配到速耐力根智或者鸽了

        for i, card in enumerate(cards):
            prob = self.character_status.composition_information.support_card_probs_lists[i]
            result = random.choices(np.arange(6), weights=prob)
            support_cards_distribution[result, i] = 1

        self.character_status.support_cards_distribution = support_cards_distribution

    def step(self, action_code):
        # print(self.character_status.turn_count)
        # print('当前体力：%s' % self.character_status.energy)
        done = False  # 是否结束
        if_success, effect = self.action_select(action_code)
        reward = self.calculate_reward(if_success, effect)
        self.total_rewards += reward
        # print('reward：%s' % reward)

        add_stats(self.character_status, effect)
        update_facilities(if_success, action_code, self.character_status)
        add_cards_bond_v2(if_success, action_code, self.character_status)

        if self.print_each_actions:
            print(f'当前回合数:{self.character_status.turn_count}')
            print(f'当前支援角色分布:\n{self.character_status.support_cards_distribution}')
            success_str = '成功' if if_success else '失败'
            print(f'当前动作选择:{action_code},{success_str},剩余体力值:{self.character_status.energy}')
            print(f'属性变动:{effect}')
            print(f'当前角色属性:{self.character_status.stats}\n')



        # print(self.character_status.train_level_list)
        # print(self.character_status.stats)
        # TODO: 行动后事件，例如出行、育成结束后事件等
        self.character_status.turn_count += 1
        # TODO: 行动前事件，大部分支援卡的随机事件
        self.random_distribution_support_cards()
        # print(self.character_status.support_cards_distribution)
        status_to_net = self.get_network()
        if self.character_status.turn_count >= 77:
            if self.log_every_game:
                print(f"最终属性:{self.character_status.stats} Total Rewards:{self.total_rewards}")
            done = True

        return status_to_net, reward, done, False, {}

    def action_select(self, action_code):
        effect = np.array([0, 0, 0, 0, 0, 0])
        if_success = False
        match action_code:
            case 0:  # 速度训练
                if_success, effect = go_training(0, self.character_status)
            case 1:  # 耐力训练
                if_success, effect = go_training(1, self.character_status)
            case 2:  # 力量训练
                if_success, effect = go_training(2, self.character_status)
            case 3:  # 根性训练
                if_success, effect = go_training(3, self.character_status)
            case 4:  # 智力训练
                if_success, effect = go_training(4, self.character_status)
            case 5:  # 休息
                rest(self.character_status)
                if_success = True
            case 6:  # 出行
                go_out(self.character_status)
                if_success = True
            case 7:  # 医疗
                return if_success, effect
            case 8:  # 比赛
                return if_success, effect
        return if_success, effect

    def calculate_reward(self, if_success, effect):
        result = 0.0
        for i, stat in enumerate(effect):
            # if i != 4:  # 降低智力的奖励
            if i != len(effect) - 1:
                # if self.character_status.stats[i] + effect[i] < 0:
                #     result -= 2 * self.character_status.stats[i]
                # el
                # 不考虑训练失败属性扣到0的情况，防止摆烂
                if self.character_status.stats[i] + effect[i] <= self.character_status.stats_upper_bound[i]:
                    result += 2 * stat
                else:
                    result += 2 * (self.character_status.stats_upper_bound[i] - self.character_status.stats[i])
            else:
                result += stat
        # if not if_success:  # 给训练失败本身加惩罚，防止0属性时摆烂
        #     result = result - 10
        # if 0 <= result < 40:  # 减少低收益训练和休息的价值
        #     result = -1
        # elif result < 0:  # 放大训练失败的惩罚
        #     result = result * 2
        return result

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        self.init_status()
        self.random_distribution_support_cards()

        net = self.get_network()
        self.set_space(net)
        return net, {}

    def print_current_status(self):
        print(self.character_status.stats)

    @property
    def max_steps(self):
        return self._max_steps

    def get_network(self):
        return neural_network.get_specific_observation_data(self.character_status, self.observation_space_type)

    def init_status(self):
        self.character_status = Status(Character('silence_suzuka'),
                                       CompositionInformation(['种马一号', '种马一号'],
                                                              ['うらら～な休日', '迫る熱に押されて',
                                                               'はやい！うまい！はやい！', 'ロード·オブ·ウオッカ',
                                                               '感謝は指先まで込めて', '一粒の安らぎ']))
        self.total_rewards = 0

    def set_space(self, net):
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(0, 1, shape=(net.shape[0],), dtype=np.float64)
