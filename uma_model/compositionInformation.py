import copy

import numpy as np

from data.parents_data import parents_data
from data.support_cards_data import support_card_data


class CompositionInformation:
    def __init__(self, parents_names, support_card_names):
        self.parents = list(map(lambda x: parents_data[x], parents_names))  # 包含两个种马信息的list
        self.support_card_list = list(map(lambda x: copy.deepcopy(support_card_data[x]), support_card_names))  # 包含六张支援卡信息的list
        self.support_card_probs_lists = []
        self.build_support_card_probs_lists()

    def build_support_card_probs_lists(self):
        list = []
        cards = self.support_card_list
        for card in cards:
            if card['card_type'] == 5 or card['card_type'] == 6:
                item = [100, 100, 100, 100, 100, 100]
            else:
                item = [100, 100, 100, 100, 100, 50]
                item[card['card_type']] *= card['specialty_rate_up'] / 100 + 1
            list.append(item)
        self.support_card_probs_lists = list
