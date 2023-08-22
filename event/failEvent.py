import numpy as np


def deal_training_failure(train_type, fail_rate, character_status):
    # TODO: 后续根据policy函数选择，现在默认选择上选项
    result = np.array([0, 0, 0, 0, 0, 0])
    if fail_rate > 20:
        # TODO： 更新心情和体力的操作可能需要用lambda传到main_loop进行，不然计算value时会有问题
        character_status.add_motivation(-3)
        character_status.add_energy(10)
        result[train_type] -= 10
        # for i in np.random.choice(5, 2):
        #     result[i] -= 10
        # if np.random.random() > 0.5: # 概率不确定
        #     # TODO: 练习down的debuff
        #     character_status.debuff[0] = 1

        # 降低随机性，减少训练难度
        result -=4

        return result
    else:
        # TODO： 更新心情和体力的操作可能需要用lambda传到main_loop进行，不然计算value时会有问题
        character_status.add_motivation(-1)
        result[train_type] -= 5
        if np.random.random() > 0.8:  # 概率不确定
            # TODO: 练习down的debuff
            character_status.debuff[0] = 1
        return result
