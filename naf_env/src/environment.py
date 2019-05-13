import math
from gym import spaces
import numpy as np

class Env:
    def __init__(self):
        high = np.array([300, 300])
        low = - high

        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.goal = [0, 0]

    def calc_shaped_reward(self, state):
        reward = 0
        done = False
        distx = self.goal[0] - state[0][0]
        disty = self.goal[1] - state[0][1]
        dist = math.sqrt(distx**2 + disty**2)

        if state[0][1] > 0.2 or state[0][1] < -0.24:
            reward += -10
            #done = True
        elif state[0][0] > 0.075 or state[0][0] < -0.35:
            reward += -10
            #done = True
        else:
            if dist < 0.005:
                reward += 100000
                done = True
            else:
                reward += -dist

        return reward, done

    def reset(self):
        pass
