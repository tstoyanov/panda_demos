import rospy
from rl_task_plugins.msg import DesiredErrorDynamicsMsg
from rl_task_plugins.msg import StateMsg
from rl_task_plugins.msg import StateMsgWithCorners
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
import csv

from hiqp_msgs.srv import *
from hiqp_msgs.msg import *
from controller_manager_msgs.srv import *
from trajectory_msgs.msg import *


class CabinetEnv(gym.Env):
    """Cabinet Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self, bEffort=True):
        super(CabinetEnv, self).__init__()

        self.goal = np.array([0, 0.1, 0.7])
        # axis defined in yz plane
        self.corners_axis = np.array([[-0.39,0.78],
                                      [-0.25,0.78],
                                      [-0.25,0.58],
                                      [-0.39,0.58]])

        self.bEffort = bEffort
        self.bViolated = False
        self.constraint_violations = 0
        self.switch_phase_done = False
        self.constraint_phase = 1
        self.reward = 0
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
        obs_low = np.array([-2.9, -2.9, -2.9, -2.9, -2.9, -2.9, -2.9,#q
                            -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,#dq
                            -1.0, -1.0, -1.0])                       #e
        self.observation_space = spaces.Box(low=obs_low, high=-obs_low, dtype=np.float32)
        
        self.action_scale = 1.0
        self.kd = 10
        
        rospy.init_node('DRL_node', anonymous=True)
        #queue_size = None forces synchronous publishing
        self.pub = rospy.Publisher('/ee_rl/act', DesiredErrorDynamicsMsg, queue_size=None)
        self.effort_pub = rospy.Publisher('/position_joint_trajectory_controller/command', JointTrajectory, queue_size=1)
        self.velocity_pub = rospy.Publisher('/velocity_joint_trajectory_controller/command', JointTrajectory, queue_size=1)
        self.rate = rospy.Rate(10)
        
        #queue size = 1 only keeps most recent message
        self.sub = rospy.Subscriber("/ee_rl/state", StateMsgWithCorners, self._next_observation, queue_size=1)
        #monitor constraints      
        self.sub_monitor = rospy.Subscriber("/hiqp_joint_velocity_controller/task_measures", TaskMeasures, self._constraint_monitor, queue_size=1)

        self.rate.sleep()
        time.sleep(1) #wait for ros to start up

        self.fresh=False

        csv_train = open("/home/quantao/panda_logs/constraints.csv", 'w', newline='')
        self.twriter = csv.writer(csv_train, delimiter=' ')
             
    def set_scale(self,action_scale):
        self.action_scale = action_scale

    def set_kd(self,kd):
        self.kd = kd    
        
    def switch_constraint_phase(self):
        # consider joint velocity controller
        if self.constraint_phase == 1:
            remove_tasks = rospy.ServiceProxy('/hiqp_joint_velocity_controller/remove_tasks', RemoveTasks)
            remove_tasks(['cage_down_corner1', 'cage_up_corner1','cage_front_corner1','cage_back_corner1','cage_left_corner1','cage_right_corner1',
                          'cage_down_corner2', 'cage_up_corner2','cage_front_corner2','cage_back_corner2','cage_left_corner2','cage_right_corner2',
                          'cage_down_corner3', 'cage_up_corner3','cage_front_corner3','cage_back_corner3','cage_left_corner3','cage_right_corner3',
                          'cage_down_corner4', 'cage_up_corner4','cage_front_corner4','cage_back_corner4','cage_left_corner4','cage_right_corner4',
                          'approach_align_x'])

        elif self.constraint_phase == 2:
            remove_tasks = rospy.ServiceProxy('/hiqp_joint_velocity_controller/remove_tasks', RemoveTasks)
            remove_tasks(['ee_cage_down', 'ee_cage_up','ee_cage_front','ee_cage_back','ee_cage_left','ee_cage_right'])
            
        remove_all_primitives = rospy.ServiceProxy('/hiqp_joint_velocity_controller/remove_all_primitives', RemoveAllPrimitives)
        remove_all_primitives()
        
        self.set_primitives()
        self.set_tasks()
    
    def set_primitives(self):
        if self.bEffort:
            hiqp_primitve_srv = rospy.ServiceProxy('/hiqp_joint_effort_controller/set_primitives', SetPrimitives)
        else:
            hiqp_primitve_srv = rospy.ServiceProxy('/hiqp_joint_velocity_controller/set_primitives', SetPrimitives)

        if self.constraint_phase == 1:
            ee_prim = Primitive(name='ee_point',type='point',frame_id='panda_hand',visible=False,color=[1,0,0,1],parameters=[0,0,0.1])
            down_plane = Primitive(name='down_plane',type='plane',frame_id='world',visible=True,color=[0,0,1,0.5],parameters=[0,0,1,0.0])
            up_plane = Primitive(name='up_plane',type='plane',frame_id='world',visible=True,color=[0,0,0.1,0.1],parameters=[0,0,1,0.9])
            back_plane = Primitive(name='back_plane',type='plane',frame_id='world',visible=True,color=[0,0,0.1,0.1],parameters=[0,1,0,-0.1])
            front_plane = Primitive(name='front_plane',type='plane',frame_id='world',visible=True,color=[0,0,0.1,0.1],parameters=[0,1,0,0.2])
            left_plane = Primitive(name='left_plane',type='plane',frame_id='world',visible=True,color=[0,0,0.1,0.1],parameters=[1,0,0,-0.2])
            right_plane = Primitive(name='right_plane',type='plane',frame_id='world',visible=True,color=[0,0,0.1,0.1],parameters=[1,0,0,0.2])         
            #hiqp_primitve_srv([ee_prim, down_plane, up_plane, back_plane, front_plane, left_plane, right_plane])

        elif self.constraint_phase == 2:
            ee_prim = Primitive(name='ee_point',type='point',frame_id='panda_hand',visible=False,color=[1,0,0,1],parameters=[0,0,0.1])
            down_plane = Primitive(name='down_plane',type='plane',frame_id='world',visible=True,color=[0,0,1,0.5],parameters=[0,0,1,0.56])
            up_plane = Primitive(name='up_plane',type='plane',frame_id='world',visible=True,color=[0,0,0.1,0.1],parameters=[0,0,1,0.8])
            back_plane = Primitive(name='back_plane',type='plane',frame_id='world',visible=True,color=[0,0,0.1,0.1],parameters=[0,1,0,-0.45])
            front_plane = Primitive(name='front_plane',type='plane',frame_id='world',visible=True,color=[0,0,0.1,0.1],parameters=[0,1,0,0.2])
            left_plane = Primitive(name='left_plane',type='plane',frame_id='world',visible=True,color=[0,0,0.1,0.1],parameters=[1,0,0,-0.2])
            right_plane = Primitive(name='right_plane',type='plane',frame_id='world',visible=True,color=[0,0,0.1,0.1],parameters=[1,0,0,0.2])
            
        book_corner1 = Primitive(name='book_corner1',type='point',frame_id='book_base_link',visible=True,color=[0,1,0,1],parameters=[0,0.07,0.1])
        book_corner2 = Primitive(name='book_corner2',type='point',frame_id='book_base_link',visible=True,color=[0,1,0,1],parameters=[0,-0.07,0.1])
        book_corner3 = Primitive(name='book_corner3',type='point',frame_id='book_base_link',visible=True,color=[0,1,0,1],parameters=[0,-0.07,-0.1])
        book_corner4 = Primitive(name='book_corner4',type='point',frame_id='book_base_link',visible=True,color=[0,1,0,1],parameters=[0,0.07,-0.1])
        corner1_x_axis = Primitive(name='corner1_x_axis',type='line',frame_id='world',visible=True,color=[1,0,0,1],parameters=[1,0,0,0,self.corners_axis[0,0],self.corners_axis[0,1]])
        corner2_x_axis = Primitive(name='corner2_x_axis',type='line',frame_id='world',visible=True,color=[1,0,0,1],parameters=[1,0,0,0,self.corners_axis[1,0],self.corners_axis[1,1]])
        corner3_x_axis = Primitive(name='corner3_x_axis',type='line',frame_id='world',visible=True,color=[1,0,0,1],parameters=[1,0,0,0,self.corners_axis[2,0],self.corners_axis[2,1]])
        corner4_x_axis = Primitive(name='corner4_x_axis',type='line',frame_id='world',visible=True,color=[1,0,0,1],parameters=[1,0,0,0,self.corners_axis[3,0],self.corners_axis[3,1]])
        book_x_axis = Primitive(name='book_x_axis',type='line',frame_id='book_base_link',visible=True,color=[1,0,0,1],parameters=[1,0,0,0,0,0])
            
        hiqp_primitve_srv([ee_prim, down_plane, up_plane, back_plane, front_plane, left_plane, right_plane,
                           book_corner1, book_corner2, book_corner3, book_corner4, 
                           corner1_x_axis, corner2_x_axis, corner3_x_axis, corner4_x_axis, book_x_axis])
        
    def set_tasks(self):
        if self.bEffort:
            hiqp_task_srv = rospy.ServiceProxy('/hiqp_joint_effort_controller/set_tasks', SetTasks)
        else:
            hiqp_task_srv = rospy.ServiceProxy('/hiqp_joint_velocity_controller/set_tasks', SetTasks)
        
        if self.constraint_phase == 1: 
            print("===>Enter Constraint Phase 1!")          
            cage_down = Task(name='ee_cage_down',priority=0,visible=True,active=True,monitored=True,
                            def_params=['TDefGeomProj','point', 'plane', 'ee_point > down_plane'],
                            dyn_params=['TDynPD', '1.0', '2.0'])
            cage_up = Task(name='ee_cage_up',priority=0,visible=True,active=True,monitored=True,
                            def_params=['TDefGeomProj','point', 'plane', 'ee_point < up_plane'],
                            dyn_params=['TDynPD', '1.0', '2.0'])
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
            rl_task = Task(name='ee_rl', priority=1, visible=True, active=True, monitored=True,
                           def_params=['TDefRLPutBook','1','0','0','0','1','0','0','0','1','ee_point','book_corner1','book_corner2','book_corner3','book_corner4'],
                           dyn_params=['TDynAsyncPolicyPutBook', '{}'.format(self.kd), 'ee_rl/act', 'ee_rl/state'])
            redundancy = Task(name='full_pose', priority=2, visible=True, active=True, monitored=True,
                          def_params=['TDefFullPose', '0.0', '-1.17', '0.0', '-2.85', '0.0', '1.82', '0.84'],
                          dyn_params=['TDynPD', '1.0', '2.0'])
            
            hiqp_task_srv([cage_down, cage_up, cage_front, cage_back, cage_left, cage_right, rl_task, redundancy])
            
        elif self.constraint_phase == 2:
            print("===>Enter Constraint Phase 2!")
            cage_down_corner1 = Task(name='cage_down_corner1',priority=0,visible=True,active=True,monitored=True,
                                     def_params=['TDefGeomProj','point', 'plane', 'book_corner1 > down_plane'],
                                     dyn_params=['TDynPD', '1.0', '2.0'])
            cage_up_corner1 = Task(name='cage_up_corner1',priority=0,visible=True,active=True,monitored=True,
                                   def_params=['TDefGeomProj','point', 'plane', 'book_corner1 < up_plane'],
                                   dyn_params=['TDynPD', '1.0', '2.0'])
            cage_front_corner1 = Task(name='cage_front_corner1',priority=0,visible=True,active=True,monitored=True,
                                      def_params=['TDefGeomProj','point', 'plane', 'book_corner1 < front_plane'],
                                      dyn_params=['TDynPD', '1.0', '2.0'])
            cage_back_corner1 = Task(name='cage_back_corner1',priority=0,visible=True,active=True,monitored=True,
                                     def_params=['TDefGeomProj','point', 'plane', 'book_corner1 > back_plane'],
                                     dyn_params=['TDynPD', '1.0', '2.0'])
            cage_left_corner1 = Task(name='cage_left_corner1',priority=0,visible=True,active=True,monitored=True,
                                     def_params=['TDefGeomProj','point', 'plane', 'book_corner1 > left_plane'],
                                     dyn_params=['TDynPD', '1.0', '2.0'])
            cage_right_corner1 = Task(name='cage_right_corner1',priority=0,visible=True,active=True,monitored=True,
                                      def_params=['TDefGeomProj','point', 'plane', 'book_corner1 < right_plane'],
                                      dyn_params=['TDynPD', '1.0', '2.0'])
            cage_down_corner2 = Task(name='cage_down_corner2',priority=0,visible=True,active=True,monitored=True,
                                     def_params=['TDefGeomProj','point', 'plane', 'book_corner2 > down_plane'],
                                     dyn_params=['TDynPD', '1.0', '2.0'])
            cage_up_corner2 = Task(name='cage_up_corner2',priority=0,visible=True,active=True,monitored=True,
                                   def_params=['TDefGeomProj','point', 'plane', 'book_corner2 < up_plane'],
                                   dyn_params=['TDynPD', '1.0', '2.0'])
            cage_front_corner2 = Task(name='cage_front_corner2',priority=0,visible=True,active=True,monitored=True,
                                      def_params=['TDefGeomProj','point', 'plane', 'book_corner2 < front_plane'],
                                      dyn_params=['TDynPD', '1.0', '2.0'])
            cage_back_corner2 = Task(name='cage_back_corner2',priority=0,visible=True,active=True,monitored=True,
                                     def_params=['TDefGeomProj','point', 'plane', 'book_corner2 > back_plane'],
                                     dyn_params=['TDynPD', '1.0', '2.0'])
            cage_left_corner2 = Task(name='cage_left_corner2',priority=0,visible=True,active=True,monitored=True,
                                     def_params=['TDefGeomProj','point', 'plane', 'book_corner2 > left_plane'],
                                     dyn_params=['TDynPD', '1.0', '2.0'])
            cage_right_corner2 = Task(name='cage_right_corner2',priority=0,visible=True,active=True,monitored=True,
                                      def_params=['TDefGeomProj','point', 'plane', 'book_corner2 < right_plane'],
                                      dyn_params=['TDynPD', '1.0', '2.0'])
            cage_down_corner3 = Task(name='cage_down_corner3',priority=0,visible=True,active=True,monitored=True,
                                     def_params=['TDefGeomProj','point', 'plane', 'book_corner3 > down_plane'],
                                     dyn_params=['TDynPD', '1.0', '2.0'])
            cage_up_corner3 = Task(name='cage_up_corner3',priority=0,visible=True,active=True,monitored=True,
                                   def_params=['TDefGeomProj','point', 'plane', 'book_corner3 < up_plane'],
                                   dyn_params=['TDynPD', '1.0', '2.0'])
            cage_front_corner3 = Task(name='cage_front_corner3',priority=0,visible=True,active=True,monitored=True,
                                      def_params=['TDefGeomProj','point', 'plane', 'book_corner3 < front_plane'],
                                      dyn_params=['TDynPD', '1.0', '2.0'])
            cage_back_corner3 = Task(name='cage_back_corner3',priority=0,visible=True,active=True,monitored=True,
                                     def_params=['TDefGeomProj','point', 'plane', 'book_corner3 > back_plane'],
                                     dyn_params=['TDynPD', '1.0', '2.0'])
            cage_left_corner3 = Task(name='cage_left_corner3',priority=0,visible=True,active=True,monitored=True,
                                     def_params=['TDefGeomProj','point', 'plane', 'book_corner3 > left_plane'],
                                     dyn_params=['TDynPD', '1.0', '2.0'])
            cage_right_corner3 = Task(name='cage_right_corner3',priority=0,visible=True,active=True,monitored=True,
                                      def_params=['TDefGeomProj','point', 'plane', 'book_corner3 < right_plane'],
                                      dyn_params=['TDynPD', '1.0', '2.0'])
            cage_down_corner4 = Task(name='cage_down_corner4',priority=0,visible=True,active=True,monitored=True,
                                     def_params=['TDefGeomProj','point', 'plane', 'book_corner4 > down_plane'],
                                     dyn_params=['TDynPD', '1.0', '2.0'])
            cage_up_corner4 = Task(name='cage_up_corner4',priority=0,visible=True,active=True,monitored=True,
                                   def_params=['TDefGeomProj','point', 'plane', 'book_corner4 < up_plane'],
                                   dyn_params=['TDynPD', '1.0', '2.0'])
            cage_front_corner4 = Task(name='cage_front_corner4',priority=0,visible=True,active=True,monitored=True,
                                      def_params=['TDefGeomProj','point', 'plane', 'book_corner4 < front_plane'],
                                      dyn_params=['TDynPD', '1.0', '2.0'])
            cage_back_corner4 = Task(name='cage_back_corner4',priority=0,visible=True,active=True,monitored=True,
                                     def_params=['TDefGeomProj','point', 'plane', 'book_corner4 > back_plane'],
                                     dyn_params=['TDynPD', '1.0', '2.0'])
            cage_left_corner4 = Task(name='cage_left_corner4',priority=0,visible=True,active=True,monitored=True,
                                     def_params=['TDefGeomProj','point', 'plane', 'book_corner4 > left_plane'],
                                     dyn_params=['TDynPD', '1.0', '2.0'])
            cage_right_corner4 = Task(name='cage_right_corner4',priority=0,visible=True,active=True,monitored=True,
                                      def_params=['TDefGeomProj','point', 'plane', 'book_corner4 < right_plane'],
                                      dyn_params=['TDynPD', '1.0', '2.0'])
            approach_align_x = Task(name='approach_align_x',priority=0,visible=True,active=True,monitored=True,
                                    def_params=['TDefGeomAlign','line', 'line', 'book_x_axis = corner2_x_axis'],
                                    dyn_params=['TDynPD', '1.0', '2.0'])
            #rl_task = Task(name='ee_rl', priority=1, visible=True, active=True, monitored=True,
            #               def_params=['TDefRLPutBook','1','0','0','0','1','0','0','0','1','ee_point','book_corner1','book_corner2','book_corner3','book_corner4'],
            #               dyn_params=['TDynAsyncPolicyPutBook', '{}'.format(self.kd), 'ee_rl/act', 'ee_rl/state'])
            #redundancy = Task(name='full_pose', priority=2, visible=True, active=True, monitored=True,
            #                  def_params=['TDefFullPose', '0.0', '-1.17', '0.0', '-2.85', '0.0', '1.82', '0.84'],
            #                  dyn_params=['TDynPD', '1.0', '2.0'])
            
            hiqp_task_srv([cage_up_corner1, cage_down_corner1, cage_front_corner1, cage_back_corner1, cage_left_corner1, cage_right_corner1,
                           cage_up_corner2, cage_down_corner2, cage_front_corner2, cage_back_corner2, cage_left_corner2, cage_right_corner2,
                           cage_up_corner3, cage_down_corner3, cage_front_corner3, cage_back_corner3, cage_left_corner3, cage_right_corner3,
                           cage_up_corner4, cage_down_corner4, cage_front_corner4, cage_back_corner4, cage_left_corner4, cage_right_corner4,
                           approach_align_x])
            
    
    def _next_observation(self, data):
        self.e = np.array(data.e)
        self.de = np.array(data.de)
        self.J = np.transpose(np.reshape(np.array(data.J_lower), [data.n_joints,data.n_constraints_lower]))       
        self.A = np.transpose(np.reshape(np.array(data.J_upper), [data.n_joints,data.n_constraints_upper]))   
        self.b = -np.reshape(np.array(data.b_upper), [data.n_constraints_upper,1])
        self.rhs = -np.reshape(np.array(data.rhs_fixed_term), [data.n_constraints_lower,1])
        self.q = np.reshape(np.array(data.q), [data.n_joints,1])
        self.dq = np.reshape(np.array(data.dq), [data.n_joints,1])
        self.ddq_star = np.reshape(np.array(data.ddq_star), [data.n_joints,1])
    
        self.J = self.J[:,:-2]
        self.A = self.A[:,:-2]
        self.q = self.q[:-2]
        self.dq = self.dq[:-2]
        self.ddq_star = self.ddq_star[:-2]

        self.book_corners = np.transpose(np.reshape(np.array(data.book_corners), [3, 4]))
        
        if all(self.book_corners[:,2]>0.555):
            self.constraint_phase = 2
        
        self.observation = np.concatenate([np.squeeze(self.q), np.squeeze(self.dq), self.e])
        
        self.fresh = True

    def _constraint_monitor(self, data):
        if True:
        #if self.switch_phase_done:
            violate_thre = 0.1
            penalty_scale = 1.0

            for task in data.task_measures:
                if task.task_name in ["cage_down_corner1", "cage_back_corner1", "cage_left_corner1", 
                                      "cage_down_corner2", "cage_back_corner2", "cage_left_corner2",
                                      "cage_down_corner3", "cage_back_corner3", "cage_left_corner3",
                                      "cage_down_corner4", "cage_back_corner4", "cage_left_corner4"] and task.e[0] < 0 and np.abs(task.e[0]) > violate_thre:
                    #print("******Constraint {} violated!******".format(task.task_name))
                    self.bViolated = True
                    self.constraint_violations += 1
                    #self.reward -= penalty_scale*np.abs(task.e[0])
                        
                if task.task_name in ["cage_up_corner1", "cage_front_corner1", "cage_right_corner1",
                                      "cage_up_corner2", "cage_front_corner2", "cage_right_corner2",
                                      "cage_up_corner3", "cage_front_corner3", "cage_right_corner3",
                                      "cage_up_corner4", "cage_front_corner4", "cage_right_corner4"] and task.e[0] > 0 and np.abs(task.e[0]) > violate_thre:
                    #print("******Constraint {} violated!******".format(task.task_name))
                    self.bViolated = True
                    self.constraint_violations += 1
                    #self.reward -= penalty_scale*np.abs(task.e[0])
                '''
                if task.task_name in ["jnt1_limits","jnt2_limits","jnt3_limits","jnt4_limits","jnt5_limits","jnt6_limits","jnt7_limits"]:
                    if task.e[0] < -violate_thre or task.e[1] > violate_thre or task.e[2] < -violate_thre or task.e[3] > violate_thre or task.e[4] < -violate_thre or task.e[5] > violate_thre:
                        print("******Constraint {} violated!******".format(task.task_name))
                '''
                
    # Execute one time step within the environment       
    def step(self, action):
        # clip action
        a = action.numpy()[0]
        if not all(np.abs(a)<=1):
            a = np.clip(a, -1, 1)
            
        a = -a * self.action_scale
        self.pub.publish(a)
        self.fresh = False
        while not self.fresh:
            self.rate.sleep()

        self.reward, done = self.calc_shaped_reward()
        
        return self.observation, self.reward, done

    def stop(self):
        print("stop function")

        joints = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        if self.bEffort:
            remove_tasks = rospy.ServiceProxy('/hiqp_joint_effort_controller/remove_tasks', RemoveTasks)
            #remove_tasks = rospy.ServiceProxy('/hiqp_joint_effort_controller/remove_all_tasks', RemoveAllTasks)
        else:
            remove_tasks = rospy.ServiceProxy('/hiqp_joint_velocity_controller/remove_tasks', RemoveTasks)
            #remove_tasks = rospy.ServiceProxy('/hiqp_joint_velocity_controller/remove_all_tasks', RemoveAllTasks)
        remove_tasks(['ee_rl'])
        #remove_tasks()
        if self.sub is not None:
            self.sub.unregister()
        cs = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        if self.bEffort:
            resp = cs({'position_joint_trajectory_controller'},{'hiqp_joint_effort_controller'},2,True,0.1)
            self.effort_pub.publish(JointTrajectory(joint_names=joints, points=[
                JointTrajectoryPoint(positions=[0.0, -1.17, 0.0, -2.85, 0.0, 1.82, 0.84], time_from_start=rospy.Duration(4.0))]))
        else:
            resp = cs({'velocity_joint_trajectory_controller'},{'hiqp_joint_velocity_controller'},2,True,0.1)
            self.velocity_pub.publish(JointTrajectory(joint_names=joints, points=[
                JointTrajectoryPoint(positions=[0.0, -1.17, 0.0, -2.85, 0.0, 1.82, 0.84], time_from_start=rospy.Duration(4.0))]))

    def start(self):
        print("start function")
        self.bViolated = False
        self.constraint_violations = 0
        self.constraint_phase = 1
        self.reward = 0
        self.switch_phase_done = False
        print("constraint phase:", self.constraint_phase)
        cs = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        if self.bEffort:
            resp = cs({'hiqp_joint_effort_controller'},{'position_joint_trajectory_controller'},2,True,0.1)
            #hiqp_task_srv = rospy.ServiceProxy('/hiqp_joint_effort_controller/set_tasks', SetTasks)
        else:
            resp = cs({'hiqp_joint_velocity_controller'},{'velocity_joint_trajectory_controller'},2,True,0.1)
            #hiqp_task_srv = rospy.ServiceProxy('/hiqp_joint_velocity_controller/set_tasks', SetTasks)
        #rl_task = Task(name='ee_rl',priority=2,visible=True,active=True,monitored=True,
        #               def_params=['TDefRL2DSpace','1','0','0','0','1','0','ee_point'],
        #               dyn_params=['TDynAsyncPolicy', '{}'.format(self.kd), 'ee_rl/act', 'ee_rl/state'])
        #hiqp_task_srv([rl_task])
        
        #self.set_primitives()# need to be checked???
        #self.set_tasks()
        self.switch_constraint_phase()
        
        #wait for fresh state
        self.fresh = False
        #queue size = 1 only keeps most recent message
        self.sub = rospy.Subscriber("/ee_rl/state", StateMsgWithCorners, self._next_observation, queue_size=1)

        while not self.fresh:
            self.rate.sleep()
        return self.observation  # reward, done, info can't be included      


    # reset through full pose
    def reset(self):
        # Reset the state of the environment to an initial state
        if not self.bEffort:
            return self.reset_vel()
        
        hiqp_deactivate_task_srv = rospy.ServiceProxy('/hiqp_joint_effort_controller/deactivate_task', DeactivateTask)
        hiqp_activate_task_srv = rospy.ServiceProxy('/hiqp_joint_effort_controller/activate_task', ActivateTask)
        
        hiqp_deactivate_task_srv('ee_rl')  
        time.sleep(10)
        hiqp_activate_task_srv('ee_rl')
        
        self.fresh = False
        while not self.fresh:
            self.rate.sleep()

        return self.observation  # reward, done, info can't be included        
         
    def render(self, mode='human'):
        self.twriter.writerow(self.A.tolist())
        self.twriter.writerow(self.J.tolist())
        self.twriter.writerow(self.b.tolist())
        #a feasible point that is the least-squares solution
        feasible_point = self.J.dot(np.linalg.pinv(self.A).dot(self.b))
        #iterating through all higher-level constraints
        n_jnts = np.shape(self.A[1])[0]
        for i in range(np.shape(self.A)[0]):
            row = self.A[i,:]
            #pseudoinverse of a matrix with linearly independent rows is A'*(AA')^-1
            pinv_row = np.reshape(np.transpose(row)/(row.dot(np.transpose(row))),[1,n_jnts])
            #point on the constraint
            bi = np.asscalar(self.b[i])
            point = self.J.dot(np.transpose(bi*pinv_row))
            #nullspace projection of constraint
            Proj = np.identity(n_jnts) - np.multiply(np.transpose(np.repeat(pinv_row,n_jnts,axis=0)),row)
            U,S,V = np.linalg.svd(self.J.dot(Proj))
            normal = np.asscalar(np.sign(U[:,1].dot(feasible_point-point)))*U[:,1]
            normal = normal/np.linalg.norm(normal)
            self.twriter.writerow(point.tolist())
            self.twriter.writerow(normal.tolist())

    def close (self):
        pass

    def calc_shaped_reward(self):
        reward = 0
        done = False

        dist = np.sum(np.linalg.norm(self.book_corners[:,1:]-self.corners_axis, axis=1))
        if dist < 0.1:
            reward += 500
            print("--- Goal reached!!! ---")
            done = True
        else:
            reward += -10*dist
            done = False
            
        return reward, done


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

