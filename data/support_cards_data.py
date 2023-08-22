import numpy as np

support_card_data = {
    'うらら～な休日': {
        'name': 'うらら～な休日',
        'unique_effect': 1,  # TODO：固有加成类型
        'card_type': 3,  # 支援卡类型，0-6分别为速耐力根智友人团队
        'friendship_bonus': 32,  # 友情加成
        'motivation_bonus': 30,  # 干劲效果提升
        'train_effect_bonus': 15,  # 训练效果提升
        'initial_stats_up': np.array([0, 0, 0, 35, 0, 0]),  # 初始属性加成，速耐力根智PT
        'race_bonus': 10,  # 赛后加成
        'fan_count_bonus': 10,  # 粉丝数加成
        'hint_rate_up': 0,  # hint率加成
        'hint_lv_bonus': 0,  # hint等级提升
        'specialty_rate_up': 50,  # 得意率
        'stats_bonus': np.array([0, 0, 0, 0, 0, 1]),  # 训练属性加成，速耐力根智PT
        'wisdom_training_recovery_up': 0,  # 智力训练回体加成
        'starting_bond_up': 0,  # 初始羁绊
        'failure_rate_down': 0,  # 失败率下降
        'energy_discount': 0,  # 体力消耗下降
        'event_recovery_amount_up': 0,  # 事件回体上升
        'event_effect_up': 0,  # 事件效果上升
    },
    '迫る熱に押されて': {
        'name': '迫る熱に押されて',
        'unique_effect': 2,  # TODO：固有加成类型
        'card_type': 0,  # 支援卡类型，0-6分别为速耐力根智友人团队
        'friendship_bonus': 25,  # 友情加成
        'motivation_bonus': 30,  # 干劲效果提升
        'train_effect_bonus': 15,  # 训练效果提升
        'initial_stats_up': np.array([0, 0, 0, 0, 0, 0]),  # 初始属性加成，速耐力根智PT
        'race_bonus': 5,  # 赛后加成
        'fan_count_bonus': 15,  # 粉丝数加成
        'hint_rate_up': 30,  # hint率加成
        'hint_lv_bonus': 2,  # hint等级提升
        'specialty_rate_up': 100,  # 得意率
        'stats_bonus': np.array([0, 0, 1, 0, 0, 0]),  # 训练属性加成，速耐力根智PT
        'wisdom_training_recovery_up': 0,  # 智力训练回体加成
        'starting_bond_up': 35,  # 初始羁绊
        'failure_rate_down': 0,  # 失败率下降
        'energy_discount': 0,  # 体力消耗下降
        'event_recovery_amount_up': 0,  # 事件回体上升
        'event_effect_up': 0,  # 事件效果上升
    },
    'はやい！うまい！はやい！': {
        'name': 'はやい！うまい！はやい！',
        'unique_effect': 3,  # TODO：固有加成类型
        'card_type': 0,  # 支援卡类型，0-6分别为速耐力根智友人团队
        'friendship_bonus': 32,  # 友情加成
        'motivation_bonus': 40,  # 干劲效果提升
        'train_effect_bonus': 10,  # 训练效果提升
        'initial_stats_up': np.array([20, 0, 0, 0, 0, 0]),  # 初始属性加成，速耐力根智PT
        'race_bonus': 10,  # 赛后加成
        'fan_count_bonus': 20,  # 粉丝数加成
        'hint_rate_up': 0,  # hint率加成
        'hint_lv_bonus': 0 ,  # hint等级提升
        'specialty_rate_up': 50,  # 得意率
        'stats_bonus': np.array([1, 0, 0, 0, 0, 0]),  # 训练属性加成，速耐力根智PT
        'wisdom_training_recovery_up': 0,  # 智力训练回体加成
        'starting_bond_up': 40,  # 初始羁绊
        'failure_rate_down': 0,  # 失败率下降
        'energy_discount': 0,  # 体力消耗下降
        'event_recovery_amount_up': 0,  # 事件回体上升
        'event_effect_up': 0,  # 事件效果上升
    },
    'ロード·オブ·ウオッカ': {
        'name': 'ロード·オブ·ウオッカ',
        'unique_effect': 4,  # TODO：固有加成类型
        'card_type': 2,  # 支援卡类型，0-6分别为速耐力根智友人团队
        'friendship_bonus': 48,  # 友情加成
        'motivation_bonus': 40,  # 干劲效果提升
        'train_effect_bonus': 10,  # 训练效果提升
        'initial_stats_up': np.array([0, 0, 0, 0, 0, 0]),  # 初始属性加成，速耐力根智PT
        'race_bonus': 5,  # 赛后加成
        'fan_count_bonus': 15,  # 粉丝数加成
        'hint_rate_up': 40,  # hint率加成
        'hint_lv_bonus': 3,  # hint等级提升
        'specialty_rate_up': 85,  # 得意率
        'stats_bonus': np.array([0, 0, 1, 0, 0, 0]),  # 训练属性加成，速耐力根智PT
        'wisdom_training_recovery_up': 0,  # 智力训练回体加成
        'starting_bond_up': 25,  # 初始羁绊
        'failure_rate_down': 0,  # 失败率下降
        'energy_discount': 0,  # 体力消耗下降
        'event_recovery_amount_up': 0,  # 事件回体上升
        'event_effect_up': 0,  # 事件效果上升
    },
    '感謝は指先まで込めて': {
        'name': '感謝は指先まで込めて',
        'unique_effect': 5,  # TODO：固有加成类型
        'card_type': 4,  # 支援卡类型，0-6分别为速耐力根智友人团队
        'friendship_bonus': 37,  # 友情加成
        'motivation_bonus': 30,  # 干劲效果提升
        'train_effect_bonus': 15,  # 训练效果提升
        'initial_stats_up': np.array([0, 0, 0, 0, 0, 35]),  # 初始属性加成，速耐力根智PT
        'race_bonus': 1,  # 赛后加成
        'fan_count_bonus': 20,  # 粉丝数加成
        'hint_rate_up': 0,  # hint率加成
        'hint_lv_bonus': 0,  # hint等级提升
        'specialty_rate_up': 35,  # 得意率
        'stats_bonus': np.array([0, 0, 0, 0, 1, 0]),  # 训练属性加成，速耐力根智PT
        'wisdom_training_recovery_up': 5,  # 智力训练回体加成
        'starting_bond_up': 15,  # 初始羁绊
        'failure_rate_down': 0,  # 失败率下降
        'energy_discount': 0,  # 体力消耗下降
        'event_recovery_amount_up': 0,  # 事件回体上升
        'event_effect_up': 0,  # 事件效果上升
    },
    '一粒の安らぎ': {
        'name': '一粒の安らぎ',
        'unique_effect': 6,  # TODO：固有加成类型
        'card_type': 0,  # 支援卡类型，0-6分别为速耐力根智友人团队
        'friendship_bonus': 37,  # 友情加成
        'motivation_bonus': 0,  # 干劲效果提升
        'train_effect_bonus': 15,  # 训练效果提升
        'initial_stats_up': np.array([0, 35, 0, 0, 0, 0]),  # 初始属性加成，速耐力根智PT
        'race_bonus': 10,  # 赛后加成
        'fan_count_bonus': 20,  # 粉丝数加成
        'hint_rate_up': 0,  # hint率加成
        'hint_lv_bonus': 0,  # hint等级提升
        'specialty_rate_up': 55,  # 得意率
        'stats_bonus': np.array([0, 1, 0, 0, 0, 0]),  # 训练属性加成，速耐力根智PT
        'wisdom_training_recovery_up': 0,  # 智力训练回体加成
        'starting_bond_up': 30,  # 初始羁绊
        'failure_rate_down': 0,  # 失败率下降
        'energy_discount': 0,  # 体力消耗下降
        'event_recovery_amount_up': 0,  # 事件回体上升
        'event_effect_up': 0,  # 事件效果上升
    },
}
