from data.characters_data import character_data


class Character:
    def __init__(self, character_name):
        self.character_name = character_name
        self.character_status_list = character_data[character_name]  # character_status_list里会包含初始属性训练加成等信息
