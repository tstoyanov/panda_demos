

class Env:
    def __init__(self):
        self.actions_space = 0

    def step(self, action, state):
        desired_action = 0
        return -(state * (desired_action - action))

    def calc_reward(self):
        pass

    def seed(self, amount):
        pass

    def reset(self):
        pass
