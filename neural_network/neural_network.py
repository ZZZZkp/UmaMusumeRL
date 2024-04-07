import numpy as np

from action.goTraining import fail_rate_calculate_v2, calculate_train_effect_v2
from uma_model.status import Status


def get_network(current_state):
    net = np.array(
        [current_state.turn_count / 100, current_state.energy / 100, current_state.motivation / 4])
    net = np.concatenate(
        (net, current_state.stats / current_state.stats_upper_bound,
         current_state.train_level_list / 10))
    # net = append_character_info(net, current_state)
    # net = append_cards_info(net, current_state)
    net = append_support_cards_distribution_info(net, current_state)
    net = append_train_status(net, current_state)
    result = net.flatten()
    return result


def get_simple_network(current_state):
    # 简化状态特征网络
    net = np.array(
        [current_state.turn_count / 100, current_state.energy / 100,
         current_state.motivation / 4])
    net = np.concatenate(
        (net, current_state.stats / current_state.stats_upper_bound,
         current_state.train_level_list / 10))
    net = append_support_cards_distribution_info(net, current_state)
    result = net.flatten()
    return result


def get_specific_observation_data(current_state: Status, type: int = 0):
    net = np.array(
        [current_state.turn_count / 100, current_state.energy / 100,
         current_state.motivation / 4])
    net = np.concatenate(
        (net, current_state.stats / current_state.stats_upper_bound,
         current_state.train_level_list / 10))
    net = append_support_cards_distribution_info(net, current_state)

    if type >= 1:
        net = append_train_status(net, current_state)
    if type >= 2:
        net = append_cards_bonds(net, current_state)

    result = net.flatten()
    return result


def append_character_info(net, current_state):
    growth_rate = np.array(current_state.character.character_status_list['growth_rate'])
    race_list = current_state.character.character_status_list['race_list']
    net = np.append(net, growth_rate)
    return np.append(net, race_list)


def append_cards_info(net, current_state):
    for card in current_state.composition_information.support_card_list:
        card_info = []
        for item in list(card.values())[1:]:
            if isinstance(item, np.ndarray):
                card_info = np.concatenate((card_info, item / 100))
            else:
                card_info = np.append(card_info, item / 100)
        net = np.concatenate((net, card_info))
    return net


# 支援角色羁绊值
def append_cards_bonds(net, current_state):
    for card in current_state.composition_information.support_card_list:
        net = np.append(net, list(card.values())[14]/100)
    return net


def append_support_cards_distribution_info(net, current_state):
    distribution_array = current_state.support_cards_distribution.flatten()
    net = np.append(net, distribution_array)
    return net


# 训练数据
def append_train_status(net, current_state):
    for i in np.arange(5):
        fail_rate = fail_rate_calculate_v2(i, current_state) / 100
        train_effect = calculate_train_effect_v2(i, current_state) / 1000
        net = np.append(net, fail_rate)
        net = np.concatenate((net, train_effect))
    return net
