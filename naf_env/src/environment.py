import rospy
from rl_task_plugins.msg import DesiredErrorDynamicsMsg
from rl_task_plugins.msg import StateMsg
import subprocess
import math
import torch
import gym
from gym import spaces
import numpy as np


class ManipulateEnv(gym.Env):
    """Manipulation Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(ManipulateEnv, self).__init__()

        self.goal = [-0.3, -0.1, 0.79]
        
        self.action_space = spaces.Box(low=np.array([-300, -300, -300]), high=np.array([300, 300, 300]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-300, -300, -300]), high=np.array([300, 300, 300]), dtype=np.float32)
                  
    def init_ros(self):
        subprocess.call("~/Workspaces/catkin_ws/src/panda_demos/panda_table_launch/scripts/sim_reset_episode.sh", shell=True)
        subprocess.call("~/Workspaces/catkin_ws/src/panda_demos/panda_table_launch/scripts/sim_picking_task.sh", shell=True)
    
        rospy.init_node('DRL_node', anonymous=True)
        rospy.Subscriber("/ee_rl/state", StateMsg, self._next_observation)
        self.pub = rospy.Publisher('/ee_rl/act', DesiredErrorDynamicsMsg, queue_size=10)
        self.rate = rospy.Rate(9)
        self.rate.sleep()
     
    def _next_observation(self, data):
        self.observation = torch.Tensor(data.e).unsqueeze(0) 
    
    def step(self, action):
        # Execute one time step within the environment
        a = action.numpy()[0] * 50
        act_pub = [a[0], a[1], a[2]]
        self.pub.publish(act_pub)
        
        reward, done, obs_hit = self.calc_shaped_reward()
        return self.observation, reward, done, obs_hit
      
    def reset(self):
        # Reset the state of the environment to an initial state
        subprocess.call("~/Workspaces/catkin_ws/src/panda_demos/panda_table_launch/scripts/sim_reset_episode_fast.sh", shell=True)
        
        return self.observation  # reward, done, info can't be included
         
    def render(self, mode='human'):
        pass

    def close (self):
        pass

    def calc_dist(self):
        distx = self.goal[0] - self.observation[0][0]
        disty = self.goal[1] - self.observation[0][1]
        distz = self.goal[2] - self.observation[0][2]
        dist = math.sqrt(distx ** 2 + disty ** 2 + distz ** 2)
        return dist

    def calc_shaped_reward(self):
        reward = 0
        done = False
        obs_hit = False

        dist = self.calc_dist()

        if dist < 0.02:
            reward += 500
            print("--- Goal reached!! ---")
            done = True
        else:
            reward += -10*dist

        return reward, done, obs_hit
        
    def calc_non_shaped_reward(self):
        reward = 0
        done = False
        dist = self.calc_dist()

        if self.observation[0][1] > 0.2 or self.observation[0][1] < -0.24:
            reward += -10
            #done = True
        elif self.observation[0][0] > 0.075 or self.observation[0][0] < -0.35:
            reward += -10
            #done = True
        else:
            if dist < 0.02:
                reward += self.goal_reward
                print("--- Goal reached!! ---")
                done = True
            else:
                reward += -0.1

        return reward, done

