import numpy as np
import random

from action.goOut import go_out
from action.goTraining import go_training, update_facilities, add_cards_bond_v2, add_stats
from action.rest import rest
from uma_model.character import Character
from uma_model.compositionInformation import CompositionInformation
from uma_model.status import Status
from neural_network.neural_network import get_network, get_simple_network


class Game:
    def __init__(self, log_every_game=True):
        self._max_steps = 78
        self.character_status = Status(Character('silence_suzuka'),
                                       CompositionInformation(['种马一号', '种马一号'],
                                                              ['うらら～な休日', '迫る熱に押されて', 'はやい！うまい！はやい！', 'ロード·オブ·ウオッカ',
                                                               '感謝は指先まで込めて', '一粒の安らぎ']))
        self.random_distribution_support_cards()
        self.log_every_game = log_every_game
        # print(self.character_status.support_cards_distribution)

    def random_distribution_support_cards(self):  # TODO: 根据得意率随机分配支援卡，返回支援卡分布信息的数据，具体结构应该是二元数组
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
        # print('reward：%s' % reward)

        add_stats(self.character_status, effect)
        update_facilities(if_success, action_code, self.character_status)
        add_cards_bond_v2(if_success, action_code, self.character_status)
        # print(self.character_status.train_level_list)
        # print(self.character_status.stats)
        # TODO: 行动后事件，例如出行、育成结束后事件等
        self.character_status.turn_count += 1
        # TODO: 行动前事件，大部分支援卡的随机事件
        self.random_distribution_support_cards()
        # print(self.character_status.support_cards_distribution)
        status_to_net = get_simple_network(self.character_status)
        if self.character_status.turn_count >= 77:
            if self.log_every_game:
                print("最终属性")
                print(self.character_status.stats)
            done = True

        return status_to_net, reward, done

    def action_select(self, action_code):  # TODO: 通过policy选择行动
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
            case 6:  # 出行
                go_out(self.character_status)
            case 7:  # 医疗
                return if_success, effect
            case 8:  # 比赛
                return if_success, effect
        return if_success, effect

    def calculate_reward(self, if_success, effect):  # TODO:根据state的改变计算动作的奖励值,奖励函数应该转换增加的属性、pt
        result = 0
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

    def reset(self, log_every_game=True):
        self.__init__(log_every_game)
        return get_simple_network(self.character_status)

    def print_current_status(self):
        print(self.character_status.stats)

    @property
    def max_steps(self):
        return self._max_steps