import rospy
from rl_task_plugins.msg import DesiredErrorDynamicsMsg
from rl_task_plugins.msg import StateMsg
from hiqp_msgs.srv import RemovePrimitives
from hiqp_msgs.srv import RemovePrimitivesRequest
from hiqp_msgs.srv import SetPrimitives
from hiqp_msgs.srv import SetPrimitivesRequest
from hiqp_msgs.msg import Primitive
import subprocess
import math
import random
import torch
import gym
from gym import spaces
import numpy as np


class ManipulateEnv(gym.Env):
    """Manipulation Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(ManipulateEnv, self).__init__()

        self.goal = [-0.2, -0.0, 0.79]
        self.reset_rand_goal = True
        
        self.action_space = spaces.Box(low=np.array([-300, -300, -300]), high=np.array([300, 300, 300]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-300, -300, -300]), high=np.array([300, 300, 300]), dtype=np.float32)
        
    def init_ros(self):
        #subprocess.call("~/Workspaces/catkin_ws/src/panda_demos/panda_table_launch/scripts/sim_reset_episode.sh", shell=True)
        subprocess.call("~/Workspaces/catkin_ws/src/panda_demos/panda_table_launch/scripts/sim_picking_task.sh", shell=True)
                
        rospy.init_node('DRL_node', anonymous=True)
        rospy.Subscriber("/ee_rl/state", StateMsg, self._next_observation)
        self.pub = rospy.Publisher('/ee_rl/act', DesiredErrorDynamicsMsg, queue_size=10)
        self.rate = rospy.Rate(9)
        self.rate.sleep()
     
    def _next_observation(self, data):
        delta_x = self.goal[0] - data.e[0]
        delta_y = self.goal[1] - data.e[1]
        delta_z = self.goal[2] - data.e[2]
        self.observation = torch.Tensor([[delta_x, delta_y, delta_z]])


    def step(self, action):
        # Execute one time step within the environment
        a = action.numpy()[0] * 100
        act_pub = [a[0], a[1], a[2]]
        self.pub.publish(act_pub)
        
        reward, done, obs_hit = self.calc_shaped_reward()
        return self.observation, reward, done, obs_hit
    
    def reset_goal(self):
        rand_x = random.uniform(-0.2, 0.0)
        rand_y = random.uniform(-0.2, 0.0)
        self.rand_goal = [rand_x, rand_y, 0.79]
        
        rospy.wait_for_service('/hiqp_joint_effort_controller/remove_primitives')
        rospy.wait_for_service('/hiqp_joint_effort_controller/set_primitives')
        try:
            # remove primitives
            remove_primitive = rospy.ServiceProxy('/hiqp_joint_effort_controller/remove_primitives', RemovePrimitives)
            remove_primitive_req = RemovePrimitivesRequest()
            remove_primitive_req.names = ['goal', 'goal_point']
            remove_primitive(remove_primitive_req)
            
            # set primitives
            set_primitive = rospy.ServiceProxy('/hiqp_joint_effort_controller/set_primitives', SetPrimitives)
            
            set_primitive_req = SetPrimitivesRequest()
            set_primitive_req.primitives = [Primitive(name='goal', type='box', frame_id='world', visible=True, color=[0.0, 1.0, 0.0, 1.0], 
                                            parameters=[self.rand_goal[0], self.rand_goal[1], self.rand_goal[2], 0.04, 0.04, 0.04]),
                                Primitive(name='goal_point', type='point', frame_id='world', visible=True, color=[0.0, 0.0, 1.0, 1.0], 
                                            parameters=[self.rand_goal[0], self.rand_goal[1], self.rand_goal[2]])]
            set_primitive(set_primitive_req)
            
            self.goal = self.rand_goal
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
            
    def reset(self):
        # Reset the state of the environment to an initial state
        subprocess.call("~/Workspaces/catkin_ws/src/panda_demos/panda_table_launch/scripts/sim_reset_episode_fast.sh", shell=True)
        
        if self.reset_rand_goal:
            self.reset_goal()

        return self.observation  # reward, done, info can't be included
         
    def render(self, mode='human'):
        pass

    def close (self):
        subprocess.call("~/Workspaces/catkin_ws/src/panda_demos/panda_table_launch/scripts/sim_reset_episode_fast.sh", shell=True)

    def calc_dist(self):
        dist = math.sqrt(self.observation[0][0] ** 2 + self.observation[0][1] ** 2 + self.observation[0][2] ** 2)
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

