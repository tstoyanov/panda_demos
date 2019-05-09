import math
from gym import spaces
import numpy as np

class Env:
    def __init__(self):
        high = np.array([10, 10])
        low = - high
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.base_goal = 0
        self.ee_goal = [0, 0]

    def step(self, action, state):
        return state - action

    def calc_shaped_reward(self, state):
        dist = math.sqrt(state[0][0]**2 + state[0][1]**2)
        if dist < 1:
            reward = 100000
            done = True
        else:
            reward = -dist
            done = False
        return reward, done

    def reset(self):
        pass
