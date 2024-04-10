# train_type是训练类型
import math
import random

import numpy as np

import event.failEvent
from data.train_effect import failure_const, ura_training_basic_effect


def go_training(train_type, character_status):
    fail_rate = fail_rate_calculate_v2(train_type, character_status)
    # print('失败率：%s' % fail_rate)
    if np.random.random_integers(100) > fail_rate:
        cost_energy(train_type, character_status)
        train_effect = calculate_train_effect_v2(train_type, character_status)
        return True, train_effect
    else:
        if train_type == 4:
            # TODO： 更新体力的操作可能需要用lambda传到main_loop进行，不然计算value时会有问题
            character_status.add_energy(5)
            return False, np.array([0, 0, 0, 0, 0, 0])
        fail_effect = event.failEvent.deal_training_failure(train_type, fail_rate, character_status)
        return False, fail_effect


# def calculate_train_effect(train_type, character_status):
#     train_level = character_status.train_level_list[train_type]
#     basic_effect = np.array(ura_training_basic_effect[train_type][train_level][:6])
#     growth_rate = np.array(character_status.character.character_status_list['growth_rate'])
#     motivation_base = [-0.2, -0.1, 0, 0.1, 0.2][character_status.motivation]
#     motivation_fix = 1
#     train_effect_bonus_fix = 1
#     friendship_bonus_fix = 1
#     card_count = 0
#     for card in character_status.support_cards_distribution[train_type]:
#         basic_effect += np.array(card['stats_bonus'])
#         motivation_fix += motivation_base * card['motivation_bonus'] / 100
#         train_effect_bonus_fix += card['train_effect_bonus'] / 100
#         card_count += 1
#         if card['starting_bond_up'] >= 80 & card['card_type'] == train_type:  # TODO：以后考虑友人和团队卡
#             friendship_bonus_fix *= (1 + card['friendship_bonus'] / 100)
#
#     return np.clip(np.ceil(basic_effect * growth_rate * motivation_fix * train_effect_bonus_fix * (
#             1 + 0.05 * card_count) * friendship_bonus_fix), 0, 100)


def calculate_train_effect_v2(train_type, character_status):
    train_level = character_status.train_level_list[train_type]
    basic_effect = np.array(ura_training_basic_effect[train_type][train_level][:6])
    growth_rate = np.array(character_status.character.character_status_list['growth_rate'])
    motivation_base = [-0.2, -0.1, 0, 0.1, 0.2][character_status.motivation]
    motivation_fix = 1
    train_effect_bonus_fix = 1
    friendship_bonus_fix = 1
    card_count = 0

    for i, value in enumerate(character_status.support_cards_distribution[train_type]):
        if value == 1:
            card = character_status.composition_information.support_card_list[i]
            # TODO: 后续需要修改成只加该训练能加的属性，现在属性加成在本来+0的训练也会生效。
            basic_effect += np.array(card['stats_bonus'])
            motivation_fix += motivation_base * card['motivation_bonus'] / 100
            train_effect_bonus_fix += card['train_effect_bonus'] / 100
            card_count += 1
            if card['starting_bond_up'] >= 80 & card['card_type'] == train_type:  # TODO：以后考虑友人和团队卡
                friendship_bonus_fix *= (1 + card['friendship_bonus'] / 100)

    return np.clip(np.ceil(basic_effect * growth_rate * motivation_fix * train_effect_bonus_fix * (
            1 + 0.05 * card_count) * friendship_bonus_fix), 0, 100).astype(np.int32)


def add_stats(character_status, train_effect):
    character_status.stats += train_effect
    character_status.stats = np.clip(character_status.stats, 0, character_status.stats_upper_bound)
    return character_status


# def add_cards_bond(if_success, train_type, character_status):
#     if if_success:
#         for card in character_status.support_cards_distribution[train_type]:
#             card['starting_bond_up'] = min(100, card['starting_bond_up'] + 7)


def add_cards_bond_v2(if_success, train_type, character_status):
    if if_success and train_type < 5:
        for i, value in enumerate(character_status.support_cards_distribution[train_type]):
            if value == 1:
                card = character_status.composition_information.support_card_list[i]
                card['starting_bond_up'] = min(100, card['starting_bond_up'] + 7)


def update_facilities(if_success, train_type, character_status):
    if if_success and train_type < 5:
        character_status.train_times[train_type] += 1
        for i, times in enumerate(character_status.train_times):
            if times >= 16:
                character_status.train_level_list[i] = 4
            elif times >= 12:
                character_status.train_level_list[i] = 3
            elif times >= 8:
                character_status.train_level_list[i] = 2
            elif times >= 4:
                character_status.train_level_list[i] = 1


def fail_rate_calculate(train_type, character_status):
    train_level = character_status.train_level_list[train_type]
    const = failure_const[train_type][train_level]
    failure_rate_down = 0
    unique_effect_fd = 0
    buff_fd = 0
    for card in character_status.support_cards_distribution[train_type]:
        failure_rate_down += card['failure_rate_down']
    #     TODO: 固有减失败率修正
    # TODO: buff失败率修正
    basic_fr = np.clip(0.025 * (character_status.energy - character_status.energy_upper_bound) * (
            character_status.energy - const / 10), 0, 99)
    return np.clip(math.ceil(basic_fr * (1 - failure_rate_down / 100) * (1 - unique_effect_fd / 100)) + buff_fd, 0, 100)


def fail_rate_calculate_v2(train_type, character_status):
    train_level = character_status.train_level_list[train_type]
    const = failure_const[train_type][train_level]
    failure_rate_down = 0
    unique_effect_fd = 0
    buff_fd = 0
    for i, value in enumerate(character_status.support_cards_distribution[train_type]):
        if value == 1:
            card = character_status.composition_information.support_card_list[i]
            failure_rate_down += card['failure_rate_down']
    #     TODO: 固有减失败率修正
    # TODO: buff失败率修正
    basic_fr = np.clip(0.025 * (character_status.energy - character_status.energy_upper_bound) * (
            character_status.energy - const / 10), 0, 99)
    return np.clip(math.ceil(basic_fr * (1 - failure_rate_down / 100) * (1 - unique_effect_fd / 100)) + buff_fd, 0, 100)


def cost_energy_calculate(train_type, character_status):
    train_level = character_status.train_level_list[train_type]
    cost = ura_training_basic_effect[train_type][train_level][6]
    if train_type == 4:
        for i, value in enumerate(character_status.support_cards_distribution[train_type]):
            if value == 1:
                card = character_status.composition_information.support_card_list[i]
                cost += card['wisdom_training_recovery_up']

    return cost


def cost_energy(train_type, character_status):
    cost = cost_energy_calculate(train_type, character_status)
    character_status.add_energy(cost)
    # print('消耗体力：%s' % cost)
    return character_status
