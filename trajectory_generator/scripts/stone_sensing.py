#!/usr/bin/env python
import json
import copy
import rospy
from franka_gripper.msg import GraspActionGoal
from franka_gripper.msg import MoveActionGoal
import geometry_msgs.msg
import moveit_msgs.msg


import ast
import sys
import moveit_commander
import getopt
import os

import thread
import pandas as pd

from franka_gripper.srv import *

import matplotlib.pyplot as plt
import seaborn as sns

import rospkg
rospack = rospkg.RosPack()

def subsample(samples_list, samples_nr):
    real_sample_interval = len(samples_list)/float(samples_nr)
    current_sample_interval = int(real_sample_interval)
    sample_interval_avg = None
    sample_index = 0
    subsamples = []
    while len(subsamples) < 100:
        if len(subsamples) != 0 and sample_index >= len(samples_list):
            sample_index = len(subsamples)-1
        subsamples.append(samples_list[sample_index])
        if sample_index != 0:
            sample_interval_avg = (sample_index / float(len(subsamples)-1))
            if sample_interval_avg > real_sample_interval:
                current_sample_interval -= 1
            elif sample_interval_avg < real_sample_interval:
                current_sample_interval += 1
        sample_index += current_sample_interval
    return subsamples
        

def callback(data, args):
    sensors_data = args[0]
    sensors_data["force"]["x"].append(data.wrench.force.x)
    sensors_data["force"]["y"].append(data.wrench.force.y)
    sensors_data["force"]["z"].append(data.wrench.force.z)
    sensors_data["torque"]["x"].append(data.wrench.torque.x)
    sensors_data["torque"]["y"].append(data.wrench.torque.y)
    sensors_data["torque"]["z"].append(data.wrench.torque.z)

def listener():
    rospy.Subscriber("/panda/franka_state_controller/F_ext", geometry_msgs.msg.WrenchStamped, callback, [])

def sense_stone(input_folder="latest", tot_time_nsecs=9000000000, is_simulation=False, is_learning=True, t=None):

    sensors_data = {
        "force": {
            "x": [],
            "y": [],
            "z": []
        },
        "torque": {
            "x": [],
            "y": [],
            "z": []
        }
    }
    rospy.init_node('force_reader', anonymous=True)
    # thread.start_new_thread(listener, ())
    
    # sns.lineplot(range(len(sensors_data["force"]["x"])), sensors_data["force"]["x"])
    # plt.show()


    if is_simulation:
        print("SIMULATION MODE")
        # robot_desctiption_str = "robot_description"
        # namespace = ""
        grasp_pub = rospy.Publisher('/franka_gripper/grasp/goal',
                            GraspActionGoal, queue_size=10)
        gripper_move_pub = rospy.Publisher('/franka_gripper/move/goal',
                            MoveActionGoal, queue_size=10)
        # getting the generated trajectory data
        if not is_learning:
            package_path = rospack.get_path("trajectory_generator")
            with open(package_path + "/generated_trajectories/cpp/" + input_folder + "/trajectories.txt", 'r') as f:
                data = f.read()
    else:
        robot_desctiption_str = "/panda/robot_description"
        namespace = "panda"
        grasp_pub = rospy.Publisher('/panda/franka_gripper/grasp/goal',
                            GraspActionGoal, queue_size=10)
        gripper_move_pub = rospy.Publisher('/panda/franka_gripper/move/goal',
                            MoveActionGoal, queue_size=10)
        if not is_learning:
            with open(input_folder, 'r') as f:
                data = f.read()
    if not is_learning:
        rospy.init_node('myWriter', anonymous=True)
    rate = rospy.Rate(0.1)  # hz

    print ("Waiting for '/panda/franka_gripper/move_service' service...")
    rospy.wait_for_service('/panda/franka_gripper/move_service')
    print ("Service found!")
    try:
        move_gripper = rospy.ServiceProxy('/panda/franka_gripper/move_service', gripper_service)
    # except rospy.ServiceException, e:
    except rospy.ServiceException as e:
        print ("Service connection failed: %s", e)
    print ("Service connected!")

    moveit_commander.roscpp_initialize(sys.argv)
    group_name = "panda_arm"
    
    if is_simulation:
        robot = moveit_commander.RobotCommander()
        group = moveit_commander.MoveGroupCommander(group_name)
    else:
        robot = moveit_commander.RobotCommander(robot_description=robot_desctiption_str, ns=namespace)
        group = moveit_commander.MoveGroupCommander(group_name, robot_description=robot_desctiption_str, ns=namespace)

    grasping_point = {
        # x, y, z, w
        "orientation": [-0.9239, 0.3826, -7.02665323908824e-05, 0.0001],
        # x, y, z
        # real height
        "position": [0.110, 0.063, 0.859],
        # # testing height
        # "position": [0.110, 0.063, 0.859]
        "joints_position": [0.23256071986483895, 0.1849605811604282, 0.05070528308319614, -2.772702876833163, -0.04564475351126129, 2.956572071540546, -0.45859744564697985]
    }
    sliding_point = {
        # x, y, z, w
        "orientation": [-0.9239, 0.3826, -7.02665323908824e-05, 0.0001],
        # x, y, z
        # real height
        "position": [-0.462, -0.110, 0.859]
        # # testing height
        # "position": [-0.462, -0.110, 0.856]
    }
    sensing_waypoints = {
        # x, y, z, w
        "orientations": [
            [-0.9239, 0.3826, -7.02665323908824e-05, 0.0001],

            [0.963, -0.083, 0.168, -0.191],
            [0.844, -0.342, 0.382, -0.153],
            [-0.726, 0.604, -0.319, 0.0719],
            [-0.794, 0.568, 0.214, 0.015],
            [0.844, -0.330, -0.379,0.185],
            [0.958, -0.016, -0.219, 0.181],

            [-0.9239, 0.3826, -7.02665323908824e-05, 0.0001],
        ],
        # x, y, z
        "positions": [
            # real height
            [0.110, -0.025, 1.100],

            [0.125, 0.025, 1.120],
            [0.145, -0.025, 1.130],
            [0.125, -0.075, 1.120],
            [0.095, 0.025, 1.120],
            [0.075, -0.055, 1.130],
            [0.095, -0.075, 1.120],

            [0.110, -0.025, 1.100]

            # # testing height
            # [0.110, 0.034, 1.047],

            # [0.126, 0.050, 1.051],
            # [0.146, 0.034, 1.068],
            # [0.126, 0.022, 1.051],
            # [0.094, 0.050, 1.051],
            # [0.074, 0.034, 1.068],
            # [0.094, 0.022, 1.051],

            # [0.110, 0.034, 1.047]
        ],
        "joints_positions": [
            [0.19445120013816, -0.21487730565614868, 0.04427450973878616, -2.3955809373590724, 0.01187810182718171, 2.180555477353596, -0.5547206810216108],

            [0.3351801368290918, -0.29662777607797913, -0.11244693900398067, -2.3844817042769044, 0.4160744979646716, 1.7924010314720245, -1.3208833060871592],
            [0.3619200299155657, -0.23360565089133745, -0.12781654499551712, -2.3745498456453022, 0.978863298841051, 2.1175465971498175, -1.117479492057386],
            [0.30448947731444703, -0.12694879574524728, -0.051233081743954974, -2.3243839509474378, 0.8545456072769047, 2.48963434716984, -0.5527099020836809],
            [0.15692759708981763, -0.2920189494293214, 0.12201354729948073, -2.3580899385820357, -0.21012455755472154, 1.705037037051065, 0.0049194150333632955],
            [0.12587268760957215, -0.11777158826903292, 0.16212125076698525, -2.21868566964802, -0.8504195135672374, 1.77099568547142, -0.2145776784027563],
            [0.14789706793375182, -0.13316567956848385, 0.0641090490445283, -2.301253765842371, -0.7098326927480935, 2.3117149482243375, -0.8717583615683947],

            [0.20444709271878062, -0.2148109244173751, 0.03434020159994296, -2.395439041634496, 0.009049621077046537, 2.1804183384366924, -0.552972294800217]
        ]
    }

    raw_input("Press enter to move to the grasping point.")
    # planning in eef space
    # pose_goal = geometry_msgs.msg.Pose()
    # pose_goal.position.x = grasping_point["position"][0]
    # pose_goal.position.y = grasping_point["position"][1]
    # pose_goal.position.z = grasping_point["position"][2]
    # pose_goal.orientation.x = grasping_point["orientation"][0]
    # pose_goal.orientation.y = grasping_point["orientation"][1]
    # pose_goal.orientation.z = grasping_point["orientation"][2]
    # pose_goal.orientation.w = grasping_point["orientation"][3]
    # group.set_pose_target(pose_goal)
    # plan = group.go(wait=True)
    # group.stop()
    # group.clear_pose_targets()

    # planning in joint space
    joint_goal = group.get_current_joint_values()
    for joint_index, _ in enumerate(grasping_point["joints_position"]):
        joint_goal[joint_index] = grasping_point["joints_position"][joint_index]
    group.go(joint_goal, wait=True)
    group.stop()
        
    raw_input("Press enter to grasp the stone.")
    grasp_message = GraspActionGoal()
    grasp_message.goal.width = 0.02
    grasp_message.goal.epsilon.inner = 0.01
    grasp_message.goal.epsilon.outer = 0.01
    grasp_message.goal.speed = 0.05
    grasp_message.goal.force = 0.01
    grasp_pub.publish(grasp_message)

    pose_goal = geometry_msgs.msg.Pose()
    waypoints = []

    raw_input("Press enter to start sensing.")
    for pos in sensing_waypoints["joints_positions"]:
        joint_goal = group.get_current_joint_values()
        for joint_index, _ in enumerate(pos):
            joint_goal[joint_index] = pos[joint_index]
        group.go(joint_goal, wait=True)
        group.stop()
        sub = rospy.Subscriber("/panda/franka_state_controller/F_ext", geometry_msgs.msg.WrenchStamped, callback, [sensors_data])
        rospy.rostime.wallsleep(1)
        sub.unregister()
    # for waypoint_index, _ in enumerate(sensing_waypoints["orientations"]):

    #     waypoints = []

    #     pose_goal.position.x = sensing_waypoints["positions"][waypoint_index][0]
    #     pose_goal.position.y = sensing_waypoints["positions"][waypoint_index][1]
    #     pose_goal.position.z = sensing_waypoints["positions"][waypoint_index][2]
    #     pose_goal.orientation.x = sensing_waypoints["orientations"][waypoint_index][0]
    #     pose_goal.orientation.y = sensing_waypoints["orientations"][waypoint_index][1]
    #     pose_goal.orientation.z = sensing_waypoints["orientations"][waypoint_index][2]
    #     pose_goal.orientation.w = sensing_waypoints["orientations"][waypoint_index][3]

    #     waypoints.append(copy.deepcopy(pose_goal))

    #     (plan, fraction) = group.compute_cartesian_path(
    #                                 waypoints,   # waypoints to follow
    #                                 0.01,        # eef_step
    #                                 0.0)         # jump_threshold
    
    #     display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    #     display_trajectory.trajectory_start = robot.get_current_state()
    #     display_trajectory.trajectory.append(plan)
    #     # Publish
    #     display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
    #                                             moveit_msgs.msg.DisplayTrajectory,
    #                                             queue_size=20)
    #     display_trajectory_publisher.publish(display_trajectory);

    #     raw_input("Press enter to start sensing.")
    #     group.execute(plan, wait=True)
    #     group.stop()
    subset = subsample(samples_list=sensors_data["force"]["x"], samples_nr=100)

    raw_input("Press enter to position the arm.")
    pose_goal.position.x = sliding_point["position"][0]
    pose_goal.position.y = sliding_point["position"][1]
    pose_goal.position.z = sliding_point["position"][2]
    pose_goal.orientation.x = sliding_point["orientation"][0]
    pose_goal.orientation.y = sliding_point["orientation"][1]
    pose_goal.orientation.z = sliding_point["orientation"][2]
    pose_goal.orientation.w = sliding_point["orientation"][3]
    group.set_pose_target(pose_goal)
    plan = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()
    gripper_move_message = MoveActionGoal()
    gripper_move_message.goal.width = 0.08
    gripper_move_message.goal.speed = 0.05
    gripper_move_pub.publish(gripper_move_message)

    print("End of sensing")

if __name__ == '__main__':
    try:
        input_folder = "latest"
        is_simulation = False
        tot_time_nsecs = 20000000000  # total execution time for the trajectory in nanoseconds
        try:
            opts, args = getopt.getopt(sys.argv[1:],"i:t:s",["input=", "nanoseconds=", "simulation="])
        except getopt.GetoptError:
            print("test.py -i <input_folder> -t <trajectory_execution_time> -s")
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print("test.py -i <input_folder> -t <trajectory_execution_time>")
                sys.exit()
            elif opt in ("-i", "--input"):
                input_folder = arg
            elif opt in ("-t", "--nanoseconds"):
                tot_time_nsecs = int(arg)
            elif opt in ("-s", "--simulation"):
                is_simulation = True

        print("input_folder = " + str(input_folder))
        print("tot_time_nsecs = " + str(tot_time_nsecs))

        sense_stone(input_folder, tot_time_nsecs, is_simulation)
    except rospy.ROSInterruptException:
        pass
