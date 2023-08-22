import numpy as np


class Status:
    def __init__(self, character, composition_information):
        self.character = character
        self.composition_information = composition_information
        # TODO: 赋予初始属性，速、耐、力、根、智、PT
        self.stats = np.array(self.character.character_status_list['init_stats'])
        self.stats_upper_bound = np.array([1200, 1200, 1200, 1200, 1200, 1200])
        self.energy = 100
        self.energy_upper_bound = 100
        self.motivation = 2
        self.buff = np.array([0, 0, 0, 0, 0])
        self.debuff = np.array([0, 0, 0, 0, 0])
        self.turn_count = 0
        self.train_times = np.array([0, 0, 0, 0, 0]) # 记录训练次数
        self.train_level_list = np.array([0, 0, 0, 0, 0])
        self.support_cards_distribution = []

    def add_motivation(self, change):
        self.motivation = np.clip(self.motivation + change, 0, 4)

    def add_energy(self, change):
        self.energy = np.clip(self.energy + change, 0, self.energy_upper_bound)
