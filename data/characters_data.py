import numpy as np

character_data = {
    'silence_suzuka': {
        'growth_rate': [1.20, 1.00, 1.00, 1.10, 1.00, 1.00], # 速耐力根智PT，PT单纯作为向量对齐用
        'init_stats': [101, 84, 77, 100, 88, 120],
        'race_list': np.zeros(78, np.int8)  # 0是训练回合，1是生涯比赛回合，TODO：后面加入比赛
        # TODO：赛程安排、适应性和技能
    }
}
