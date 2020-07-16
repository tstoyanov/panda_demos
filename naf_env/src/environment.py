import rospy
from rl_task_plugins.msg import DesiredErrorDynamicsMsg
from rl_task_plugins.msg import StateMsg
import subprocess
import math
import torch
import gym
from gym import spaces
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from torch.autograd import Variable

from hiqp_msgs.srv import *
from hiqp_msgs.msg import *
from controller_manager_msgs.srv import *
from trajectory_msgs.msg import *


class ManipulateEnv(gym.Env):
    """Manipulation Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(ManipulateEnv, self).__init__()

        self.goal = [-0.2, 0.0, 0.79]
        self.reset_rand_goal = False
        
        self.action_space = spaces.Box(low=np.array([-300, -300, -300]), high=np.array([300, 300, 300]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-300, -300, -300]), high=np.array([300, 300, 300]), dtype=np.float32)
        
        subprocess.call("~/Workspaces/catkin_ws/src/panda_demos/panda_table_launch/scripts/sim_picking_task.sh", shell=True)
        
        rospy.init_node('DRL_node', anonymous=True)
        rospy.Subscriber("/ee_rl/state", StateMsg, self._next_observation)
        self.pub = rospy.Publisher('/ee_rl/act', DesiredErrorDynamicsMsg, queue_size=10)
        self.effort_pub = rospy.Publisher('/position_joint_trajectory_controller/command', JointTrajectory)
        self.rate = rospy.Rate(10)
        self.rate.sleep()
        time.sleep(1) #wait for ros to start up

     
    def set_primitives(self):
        hiqp_primitve_srv = rospy.ServiceProxy('/hiqp_joint_effort_controller/set_primitives', SetPrimitives)
        ee_prim = Primitive(name='ee_point',type='point',frame_id='world',visible=True,color=[1,0,0,1],parameters=[0,0,0])
        goal_prim = Primitive(name='goal',type='box',frame_id='world',visible=True,color=[0,1,0,1],parameters=[self.goal[0],self.goal[1],0,0.02])
        back_plane = Primitive(name='back_plane',type='plane',frame_id='world',visible=True,color=[0,1,0,0.5],parameters=[0,1,0,-0.8])
        front_plane = Primitive(name='front_plane',type='plane',frame_id='world',visible=True,color=[0,1,0,0.5],parameters=[0,1,0,0.8])
        left_plane = Primitive(name='left_plane',type='plane',frame_id='world',visible=True,color=[0,1,0,0.5],parameters=[1,0,0,-0.8])
        right_plane = Primitive(name='right_plane',type='plane',frame_id='world',visible=True,color=[0,1,0,0.5],parameters=[1,0,0,0.8])
        
        hiqp_primitve_srv([ee_prim, back_plane, front_plane, left_plane, right_plane, goal_prim])
        
    def set_tasks(self):
        hiqp_task_srv = rospy.ServiceProxy('/hiqp_joint_effort_controller/set_tasks', SetTasks)
        cage_front = Task(name='ee_cage_front',priority=0,visible=True,active=True,monitored=True,
                          def_params=['TDefGeomProj','point', 'plane', 'ee_point < front_plane'],
                          dyn_params=['TDynPD', '1.0', '2.0'])
        cage_back = Task(name='ee_cage_back',priority=0,visible=True,active=True,monitored=True,
                          def_params=['TDefGeomProj','point', 'plane', 'ee_point > back_plane'],
                          dyn_params=['TDynPD', '1.0', '2.0'])
        cage_left = Task(name='ee_cage_left',priority=0,visible=True,active=True,monitored=True,
                          def_params=['TDefGeomProj','point', 'plane', 'ee_point > left_plane'],
                          dyn_params=['TDynPD', '1.0', '2.0'])
        cage_right = Task(name='ee_cage_right',priority=0,visible=True,active=True,monitored=True,
                          def_params=['TDefGeomProj','point', 'plane', 'ee_point < right_plane'],
                          dyn_params=['TDynPD', '1.0', '2.0'])
        rl_task = Task(name='ee_rl',priority=1,visible=True,active=True,monitored=True,
                          def_params=['TDefRL2DSpace','1','0','0','0','1','0','ee_point'],
                          dyn_params=['TDynAsyncPolicy', '10.0', 'ee_rl/act', 'ee_rl/state', '/home/tsv/hiqp_logs/'])
        redundancy = Task(name='full_pose',priority=2,visible=True,active=True,monitored=True,
                          def_params=['TDefFullPose', '0.1', '-0.3', '0.0'],
                          dyn_params=['TDynPD', '0.5', '1.5'])
        
        hiqp_task_srv([cage_front, cage_back, cage_left, cage_right, rl_task, redundancy])    
    
    def _next_observation(self, data):
        delta_x = data.e[0] - self.goal[0]
        delta_y = data.e[1] - self.goal[1]
        delta_z = data.e[2] - self.goal[2]
        self.observation = torch.Tensor([[delta_x, delta_y, delta_z]])


    def step(self, action):
        # Execute one time step within the environment
        a = action.numpy()[0] * 20.0
        act_pub = [a[0], a[1], a[2]]
        self.pub.publish(act_pub)
        self.rate.sleep()
        
        reward, done, obs_hit = self.calc_shaped_reward()
        return self.observation, reward, done, obs_hit

    def reset(self):
        # Reset the state of the environment to an initial state
        #subprocess.call("~/Workspaces/catkin_ws/src/panda_demos/panda_table_launch/scripts/sim_reset_episode_fast.sh", shell=True)
        
        if self.reset_rand_goal:
            self.reset_goal()
            
        cs = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        cs_unload = rospy.ServiceProxy('/controller_manager/unload_controller', UnloadController)
        cs_load = rospy.ServiceProxy('/controller_manager/load_controller', LoadController)
        remove_tasks = rospy.ServiceProxy('/hiqp_joint_effort_controller/remove_tasks', RemoveTasks)
        #print('removing tasks')
        remove_tasks(['ee_cage_back', 'ee_cage_left', 'ee_cage_right', 'ee_cage_front', 'ee_plane_table', 'approach_align_z', 'ee_rl', 'full_pose'])
        #time.sleep(1)

        #stop hiqp
        #print('switching controller')
        resp = cs({'position_joint_trajectory_controller'}, {'hiqp_joint_effort_controller'}, 2, True, 0.1)
        #time.sleep(0.2)
        cs_unload('hiqp_joint_effort_controller')

        #print('setting to home pose')
        joints = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        self.effort_pub.publish(JointTrajectory(joint_names=joints, points=[JointTrajectoryPoint(positions=[0.1, -0.3, 0.0, 0.0, 0.1, -0.3, 0.0], time_from_start=rospy.Duration(4.0))]))
        time.sleep(4.5)
        #restart hiqp
        cs_load('hiqp_joint_effort_controller')

        #print("restarting controller")
        resp = cs({'hiqp_joint_effort_controller'}, {'position_joint_trajectory_controller'}, 2, True, 0.1)
        #set tasks to controller
        self.set_primitives()
        self.set_tasks()
        #self.pub.publish([0,0])
        time.sleep(0.5)
        #print("Now acting")

        return self.observation  # reward, done, info can't be included
    
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


    def Q_plot(self, agent, episode):
        for z in range(3):
            Q_states = []
            Q_values = []
            for i in range(0, 50):
                Q_states.append([])
                Q_values.append([])
                for j in range(0, 50):
                    delta_x = (i-40) / 100.0 - self.goal[0]
                    delta_y = (j-30) / 100.0 - self.goal[1]
                    delta_z = z * 0.1
                    state = [delta_x, delta_y, delta_z]
                    Q_state = torch.Tensor(state).unsqueeze(0)
                    Q_action = agent.select_action(Q_state)
                    agent.model.eval()
                    _, Q_value, _ = agent.model((Variable(Q_state), Variable(Q_action)))
                    agent.model.train()
        
                    Q_states[i].append(Q_state)
                    Q_values[i].append(Q_value)
        
            for k in range(len(Q_values)):
                Q_color = []
                for l in range(len(Q_values[k])):
                    Q_color.append(Q_values[k][l].data.numpy()[0][0])
                cmap = plt.cm.viridis
                #cNorm = colors.Normalize(vmin=np.min(Q_values), vmax=np.max(Q_values))
                #scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
                scalarMap = cmx.ScalarMappable(cmap=cmap)
                colorVal = scalarMap.to_rgba(Q_color)
                sx, sy, sz = torch.cat(Q_states[k]).numpy().T
                plt.scatter(sx, sy, c=colorVal, edgecolors='face')
              
            plt.xlabel('delta_x')
            plt.ylabel('delta_y')
            plt.title("Q value")
        
            fig = 'State_Q_ep_{}_z_{}_{}'.format(episode, z, '.png')
            plt.savefig(fig)
            plt.close()

