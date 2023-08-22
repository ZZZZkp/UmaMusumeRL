import numpy as np


def rest(character_status):
    rd = np.random.random()
    # TODO: 休息功能
    if rd < 0.15:
        character_status.add_energy(30)
    elif 0.15 < rd < 0.75:
        character_status.add_energy(50)
    else:
        character_status.add_energy(70)
    return character_status
