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
import csv

from hiqp_msgs.srv import *
from hiqp_msgs.msg import *
from controller_manager_msgs.srv import *
from trajectory_msgs.msg import *

from halfspaces import qhull



class ManipulateEnv(gym.Env):
    """Manipulation Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self, bEffort=True):
        super(ManipulateEnv, self).__init__()

        self.goal = np.array([0, 0, 0.72])
        self.bEffort = bEffort
        self.bConstraint = False
        self.constraint_violations = 0
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
        self.set_primitives()
        self.set_tasks()
        
        #queue size = 1 only keeps most recent message
        self.sub = rospy.Subscriber("/ee_rl/state", StateMsg, self._next_observation, queue_size=1)
        #monitor constraints      
        self.sub_monitor = rospy.Subscriber("/hiqp_joint_velocity_controller/task_measures", TaskMeasures, self._constraint_monitor, queue_size=1)

        self.rate.sleep()
        time.sleep(1) #wait for ros to start up

        self.fresh=False

        csv_train = open("/home/quantao/panda_logs/constraints.csv", 'w', newline='')
        self.twriter = csv.writer(csv_train, delimiter=' ')
        self.episode_trace = [(np.identity(self.action_space.shape[0]),self.action_space.high,0)]
             
    def set_scale(self,action_scale):
        self.action_scale = action_scale

    def set_kd(self,kd):
        self.kd = kd    
        
    def set_primitives(self):
        if self.bEffort:
            hiqp_primitve_srv = rospy.ServiceProxy('/hiqp_joint_effort_controller/set_primitives', SetPrimitives)
        else:
            hiqp_primitve_srv = rospy.ServiceProxy('/hiqp_joint_velocity_controller/set_primitives', SetPrimitives)

        ee_prim = Primitive(name='ee_point',type='point',frame_id='panda_hand',visible=True,color=[1,0,0,1],parameters=[0,0,0.1])
        #ee_prim_link6 = Primitive(name='ee_point_link6',type='point',frame_id='panda_link6',visible=True,color=[1,0,0,1],parameters=[0,0,0.1])
        goal_prim = Primitive(name='goal',type='box',frame_id='world',visible=True,color=[0,1,0,1],parameters=[self.goal[0],self.goal[1],self.goal[2],0.04, 0.04, 0.04])
        table_plane = Primitive(name='table_plane',type='plane',frame_id='world',visible=True,color=[0,0,1,0.5],parameters=[0,0,1,0.7])
        up_plane = Primitive(name='up_plane',type='plane',frame_id='world',visible=False,color=[0,0,0.1,0.1],parameters=[0,0,1,1.3])
        back_plane = Primitive(name='back_plane',type='plane',frame_id='world',visible=True,color=[0,0,0.1,0.1],parameters=[0,1,0,-0.3])
        front_plane = Primitive(name='front_plane',type='plane',frame_id='world',visible=True,color=[0,0,0.1,0.1],parameters=[0,1,0,0.3])
        left_plane = Primitive(name='left_plane',type='plane',frame_id='world',visible=True,color=[0,0,0.1,0.1],parameters=[1,0,0,-0.3])
        right_plane = Primitive(name='right_plane',type='plane',frame_id='world',visible=True,color=[0,0,0.1,0.1],parameters=[1,0,0,0.3])
        #table_z_axis = Primitive(name='table_z_axis',type='line',frame_id='world',visible=True,color=[0,1,1,1],parameters=[0,0,1,0,0,0])
        #ee_z_axis = Primitive(name='ee_z_axis',type='line',frame_id='panda_hand',visible=True,color=[0,1,1,1],parameters=[0,0,1,0,0,0])
        # 8 corners      
        corner1 = Primitive(name='corner1',type='sphere',frame_id='world',visible=True,color=[0,0,1,1],parameters=[0.3,0.3,0.7,0.01])
        corner2 = Primitive(name='corner2',type='sphere',frame_id='world',visible=True,color=[0,0,1,1],parameters=[0.3,-0.3,0.7,0.01])
        corner3 = Primitive(name='corner3',type='sphere',frame_id='world',visible=True,color=[0,0,1,1],parameters=[-0.3,-0.3,0.7,0.01])
        corner4 = Primitive(name='corner4',type='sphere',frame_id='world',visible=True,color=[0,0,1,1],parameters=[-0.3,0.3,0.7,0.01])
        corner5 = Primitive(name='corner5',type='sphere',frame_id='world',visible=True,color=[0,0,1,1],parameters=[0.3,0.3,1.3,0.01])
        corner6 = Primitive(name='corner6',type='sphere',frame_id='world',visible=True,color=[0,0,1,1],parameters=[0.3,-0.3,1.3,0.01])
        corner7 = Primitive(name='corner7',type='sphere',frame_id='world',visible=True,color=[0,0,1,1],parameters=[-0.3,-0.3,1.3,0.01])
        corner8 = Primitive(name='corner8',type='sphere',frame_id='world',visible=True,color=[0,0,1,1],parameters=[-0.3,0.3,1.3,0.01])
        
        hiqp_primitve_srv([ee_prim, goal_prim, table_plane, up_plane, back_plane, front_plane, left_plane, right_plane,
                           corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8])
        
    def set_tasks(self):
        if self.bEffort:
            hiqp_task_srv = rospy.ServiceProxy('/hiqp_joint_effort_controller/set_tasks', SetTasks)
        else:
            hiqp_task_srv = rospy.ServiceProxy('/hiqp_joint_velocity_controller/set_tasks', SetTasks)

            
        ee_plane = Task(name='ee_plane_table',priority=0,visible=True,active=True,monitored=True,
                        def_params=['TDefGeomProj','point', 'plane', 'ee_point > table_plane'],
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
        #approach_align_z = Task(name='approach_align_z',priority=0,visible=True,active=True,monitored=True,
        #                        def_params=['TDefGeomAlign','line', 'line', 'ee_z_axis = table_z_axis'],
        #                        dyn_params=['TDynPD', '1.0', '2.0'])

        rl_task = Task(name='ee_rl',priority=1,visible=True,active=True,monitored=True,
                          def_params=['TDefRLPick','1','0','0','0','1','0','0','0','1','ee_point'],
                          dyn_params=['TDynAsyncPolicy', '{}'.format(self.kd), 'ee_rl/act', 'ee_rl/state'])
        redundancy = Task(name='full_pose',priority=2,visible=True,active=True,monitored=True,
                          def_params=['TDefFullPose', '0.0', '-1.17', '0.0', '-2.85', '0.0', '1.82', '0.84'],
                          dyn_params=['TDynPD', '1.0', '2.0'])

        hiqp_task_srv([ee_plane, cage_up, cage_front, cage_back, cage_left, cage_right, rl_task, redundancy])
        #hiqp_task_srv([rl_task])
    
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
        
        self.observation = np.concatenate([np.squeeze(self.q), np.squeeze(self.dq), self.e-self.goal])
        
        self.fresh = True

    def _constraint_monitor(self, data):
        violate_thre = 0.1
        penalty_scale = 1.0
        for task in data.task_measures:
            if task.task_name == "ee_cage_back" and task.e[0] < 0:
                if np.abs(task.e[0]) > violate_thre:
                    print("*************ee_cage_back violated!******", task.e[0])
                    self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
                    self.constraint_violation += 1

                    
            if task.task_name == "ee_cage_front" and task.e[0] > 0:
                if np.abs(task.e[0]) > violate_thre:
                    print("*************ee_cage_front violated!******", task.e[0])
                    self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
                    self.constraint_violation += 1
            
            if task.task_name == "ee_cage_left" and task.e[0] < 0:
                if np.abs(task.e[0]) > violate_thre:
                    print("*************ee_cage_left violated!******", task.e[0])
                    self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
                    self.constraint_violation += 1
            
            if task.task_name == "ee_cage_right" and task.e[0] > 0:
                if np.abs(task.e[0]) > violate_thre:
                    print("*************ee_cage_right violated!******", task.e[0])
                    self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
                    self.constraint_violation += 1
                
            if task.task_name == "ee_cage_up" and task.e[0] > 0:
                if np.abs(task.e[0]) > violate_thre:
                    print("*************ee_cage_up violated!******", task.e[0])
                    self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
                    self.constraint_violation += 1
            
            if task.task_name == "ee_plane_table" and task.e[0] < 0:
                if np.abs(task.e[0]) > violate_thre:
                    print("*************ee_plane_table violated!******", task.e[0])
                    self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
                    self.constraint_violation += 1
              
            if task.task_name == "jnt1_limits":
                if task.e[0] < -violate_thre or task.e[1] > violate_thre or task.e[2] < -violate_thre or task.e[3] > violate_thre or task.e[4] < -violate_thre or task.e[5] > violate_thre:
                    print("*************jnt1_limits violated!")
                    self.bConstraint = True
                    self.constraint_violation += 1
                    
            if task.task_name == "jnt2_limits":
                if task.e[0] < -violate_thre or task.e[1] > violate_thre or task.e[2] < -violate_thre or task.e[3] > violate_thre or task.e[4] < -violate_thre or task.e[5] > violate_thre:
                    print("*************jnt2_limits violated!")
                    #self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
                    self.constraint_violation += 1
                    
            if task.task_name == "jnt3_limits":
                if task.e[0] < -violate_thre or task.e[1] > violate_thre or task.e[2] < -violate_thre or task.e[3] > violate_thre or task.e[4] < -violate_thre or task.e[5] > violate_thre:
                    print("*************jnt3_limits violated!")
                    #self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
                    self.constraint_violation += 1
             
            if task.task_name == "jnt4_limits":
                if task.e[0] < -violate_thre or task.e[1] > violate_thre or task.e[2] < -violate_thre or task.e[3] > violate_thre or task.e[4] < -violate_thre or task.e[5] > violate_thre:
                    print("*************jnt4_limits violated!")
                    #self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
                    self.constraint_violation += 1

            if task.task_name == "jnt5_limits":
                if task.e[0] < -violate_thre or task.e[1] > violate_thre or task.e[2] < -violate_thre or task.e[3] > violate_thre or task.e[4] < -violate_thre or task.e[5] > violate_thre:
                    print("*************jnt5_limits violated!")
                    #self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
                    self.constraint_violation += 1
                   
            if task.task_name == "jnt6_limits":
                if task.e[0] < -violate_thre or task.e[1] > violate_thre or task.e[2] < -violate_thre or task.e[3] > violate_thre or task.e[4] < -violate_thre or task.e[5] > violate_thre:
                    print("*************jnt6_limits violated!")
                    #self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
                    self.constraint_violation += 1
                    
            if task.task_name == "jnt7_limits":
                if task.e[0] < -violate_thre or task.e[1] > violate_thre or task.e[2] < -violate_thre or task.e[3] > violate_thre or task.e[4] < -violate_thre or task.e[5] > violate_thre:
                    print("*************jnt7_limits violated!")
                    #self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
                    self.constraint_violation += 1
            
    # Execute one time step within the environment       
    def step(self, action):
        # clip action
        a = action.numpy()[0]
        if not all(np.abs(a)<=1):
            a = np.clip(a, -1, 1)
            
        a = a * self.action_scale
        self.pub.publish(a)
        self.fresh = False
        while not self.fresh:
            self.rate.sleep()


        self.reward, done = self.calc_shaped_reward()
        
        return self.observation, self.reward, done#, Ax, bx   
    
    def project_constraint(self):   
        success, Ax, bx = qhull(self.A, self.J, self.b)

        #???
        Ax = -Ax
        if(success) :
            #self.twriter.writerow(self.episode_trace[-1][0])
            #self.twriter.writerow(self.episode_trace[-1][1])
            #self.twriter.writerow(self.A)
            #self.twriter.writerow(self.b)
            #self.twriter.writerow(self.J)
            #self.twriter.writerow(action.numpy()[0])
            #self.twriter.writerow(self.observation)
            #self.twriter.writerow(self.ddq_star)
            #self.twriter.writerow(self.rhs)
            #???
            bx = bx - Ax.dot(self.rhs).transpose()
            #bx = bx + Ax.dot(self.rhs).transpose()
            #we should be checking the actions that were feasible according to previous set of constraints
            #Axw = self.episode_trace[-1][0].dot(action.numpy()[0] * self.action_scale)
            #feasible = Axw - self.episode_trace[-1][1]
            #n_infeasible = np.sum(feasible>0.001)
            #self.episode_trace.append((Ax,bx,n_infeasible))
                 
        return Ax, bx

    def stop(self):
        self.episode_trace.clear()
        self.episode_trace = [(np.identity(self.action_space.shape[0]),self.action_space.high,0)]
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
            self.velocity_pub.publish(JointTrajectory(joint_names=joints,points=[
                JointTrajectoryPoint(positions=[0.0, -1.17, 0.0, -2.85, 0.0, 1.82, 0.84],time_from_start=rospy.Duration(4.0))]))

    def start(self):
        self.bConstraint = False
        self.constraint_violation = 0
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
        self.set_tasks()
        #wait for fresh state
        self.fresh = False
        #queue size = 1 only keeps most recent message
        self.sub = rospy.Subscriber("/ee_rl/state", StateMsg, self._next_observation, queue_size=1)

        while not self.fresh:
            self.rate.sleep()
        return self.observation  # reward, done, info can't be included      


    # reset through full pose
    def reset(self):
        # Reset the state of the environment to an initial state
        if not self.bEffort:
            return self.reset_vel()
        
        self.episode_trace.clear()
        self.episode_trace = [(np.identity(self.action_space.shape[0]),self.action_space.high,0)]
        
        hiqp_deactivate_task_srv = rospy.ServiceProxy('/hiqp_joint_effort_controller/deactivate_task', DeactivateTask)
        hiqp_activate_task_srv = rospy.ServiceProxy('/hiqp_joint_effort_controller/activate_task', ActivateTask)
        
        hiqp_deactivate_task_srv('ee_rl')  
        time.sleep(10)
        hiqp_activate_task_srv('ee_rl')
        
        self.fresh = False
        while not self.fresh:
            self.rate.sleep()

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
    
    def calc_dist(self):
        dist = np.linalg.norm(self.e-self.goal)      # math.sqrt((self.observation[0]-self.goal[0]) ** 2 + (self.observation[1]-self.goal[1])  ** 2)
        return dist

    def calc_shaped_reward(self):
        reward = 0
        done = False

        dist = self.calc_dist()
        
        if dist < 0.02:
            reward += 500
            print("--- Goal reached!! ---")
            done = True
        else:
            reward += -10*dist

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

