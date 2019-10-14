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

import argparse
import pandas as pd
import random

from franka_gripper.srv import *

import matplotlib.pyplot as plt
import seaborn as sns

import rospkg
rospack = rospkg.RosPack()
package_path = rospack.get_path("trajectory_generator")

parser = argparse.ArgumentParser(description='stone sensing')

parser.add_argument('--stones_labels', nargs='+', default=None, type=str, help='list of the labels of the stones')
parser.add_argument('--repetitions', default=2, type=int, help='number of times to sense each stone')

args, unknown = parser.parse_known_args()

def subsample_list(list_to_subsample, samples_nr):
    real_sample_interval = len(list_to_subsample)/float(samples_nr)
    current_sample_interval = int(real_sample_interval)
    sample_interval_avg = None
    sample_index = 0
    test = []
    subsamples = []
    while len(subsamples) < 100:
        if len(subsamples) != 0 and sample_index >= len(list_to_subsample):
            sample_index = len(subsamples)-1
        subsamples.append(list_to_subsample[sample_index])
        test.append(sample_index)
        if sample_index != 0:
            sample_interval_avg = (sample_index / float(len(subsamples)-1))
            if sample_interval_avg > real_sample_interval:
                current_sample_interval = int(real_sample_interval)
                # current_sample_interval -= 1
            elif sample_interval_avg < real_sample_interval:
                current_sample_interval = int(real_sample_interval) + 1
                # current_sample_interval += 1
        sample_index += current_sample_interval
    return subsamples

def filter_list(list_to_filter, alpha):
    filtered_list = []
    for index, item in enumerate(list_to_filter):
        if index == 0:
            filtered_list.append(item)
        else:
            filtered_list.append(float(alpha)*filtered_list[-1] + float(1-alpha)*item)
    return filtered_list
        

def callback(data, args):
    sensing_step = args[0]
    sensing_step["force"]["x"].append(data.wrench.force.x)
    sensing_step["force"]["y"].append(data.wrench.force.y)
    sensing_step["force"]["z"].append(data.wrench.force.z)
    sensing_step["torque"]["x"].append(data.wrench.torque.x)
    sensing_step["torque"]["y"].append(data.wrench.torque.y)
    sensing_step["torque"]["z"].append(data.wrench.torque.z)

def listener():
    rospy.Subscriber("/panda/franka_state_controller/F_ext", geometry_msgs.msg.WrenchStamped, callback, [])

def sense_stone(output_folder="latest", is_simulation=False, is_learning=True, repetitions=100, labels=None):
    package_path = rospack.get_path("trajectory_generator")
    output_folder = package_path + "/sensing_data/" + output_folder
    sensors_data = {
        "raw": {
            "force": {
                "x": [],
                "y": [],
                "z": []
            },
            "torque": {
                "x": [],
                "y": [],
                "z": []
            },
            "label": [],
            "subsamples": {
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
        },
        "filtered": {
            "alpha": None,
            "force": {
                "x": [],
                "y": [],
                "z": []
            },
            "torque": {
                "x": [],
                "y": [],
                "z": []
            },
            "subsamples": {
                "force": {
                    "x": [],
                    "y": [],
                    "z": []
                },
                "torque": {
                    "x": [],
                    "y": [],
                    "z": []
                },
                "label": []
            }
        }
    }
    rospy.init_node('force_reader', anonymous=True)

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

    grasping_points = {
        "10": {            
            "orientation": [-0.9239, 0.3826, -7.02665323908824e-05, 0.0001], # x, y, z, w
            "position": [0.360, 0.063, 0.959], # x, y, z
            "hovering_joints_position": [0.36696093194814283, 0.17944039607466314, 0.4089761023876119, -2.358461730187831, -0.12317095402943311, 2.5191082320543745, 0.08425164690794548],
            "grasping_joints_position": [0.4260575038152828, 0.4334983963393511, 0.3151125600120057, -2.3362570904182687, -0.3284292590488873, 2.7300798716132246, 0.23072858667767124]
        },
        "20": {
            "orientation": [-0.9239, 0.3826, -7.02665323908824e-05, 0.0001], # x, y, z, w
            "position": [0.210, 0.063, 0.959], # x, y, z
            "hovering_joints_position": [0.09197913135191854, -0.06794736741508771, 0.4429717659580504, -2.664749506900185, 0.05786304345829169, 2.6017944910261366, -0.30182371413154874],
            "grasping_joints_position": [0.1547609451772318, 0.2666294309465509, 0.30558009457633056, -2.639995261242515, -0.3112949253521605, 2.882298212263319, -0.0340398218951605]
        },
        "30": {
            "orientation": [-0.9239, 0.3826, -7.02665323908824e-05, 0.0001], # x, y, z, w
            "position": [0.060, 0.063, 0.959], # x, y, z
            "hovering_joints_position": [-0.2963754566866036, -0.22609519766033184, 0.5066942452590402, -2.822304789117009, 0.21546041037639288, 2.613143800150113, -0.7740535357700479],
            "grasping_joints_position": [-0.19486450364276442, 0.17763616890843223, 0.28773354355127934, -2.8004517943398994, -0.28575185754343313, 2.962733009843974, -0.4160978253429793]
        },
        "40": {
            "orientation": [-0.9239, 0.3826, -7.02665323908824e-05, 0.0001], # x, y, z, w
            "position": [-0.090, 0.063, 0.959], # x, y, z
            "hovering_joints_position": [0.2374669449747654, -0.20859305197716996, -0.5233313921309419, -2.8038381514632906, -0.20888573720058043, 2.6131777341570905, -0.8799971305608723],
            "grasping_joints_position": [0.12372470620042843, 0.18846930262097014, -0.2952061458441635, -2.781528095056179, 0.29313913822709015, 2.9528340691460504, -1.2398218304215838]
        }
    }

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
            # [0.30448947731444703, -0.12694879574524728, -0.051233081743954974, -2.3243839509474378, 0.8545456072769047, 2.48963434716984, -0.5527099020836809],
            [0.15692759708981763, -0.2920189494293214, 0.12201354729948073, -2.3580899385820357, -0.21012455755472154, 1.705037037051065, 0.0049194150333632955],
            [0.12587268760957215, -0.11777158826903292, 0.16212125076698525, -2.21868566964802, -0.8504195135672374, 1.77099568547142, -0.2145776784027563],
            # [0.14789706793375182, -0.13316567956848385, 0.0641090490445283, -2.301253765842371, -0.7098326927480935, 2.3117149482243375, -0.8717583615683947],

            [0.20444709271878062, -0.2148109244173751, 0.03434020159994296, -2.395439041634496, 0.009049621077046537, 2.1804183384366924, -0.552972294800217]
        ]
    }

    for repetition in range(repetitions):
        # if type(labels) == list and len(labels) > 1:
        #     random.shuffle(labels)
        
        for label in labels:
            # raw_input("Press enter to sense.")
            # raw_input("Press enter to move to the grasping point.")
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
            for joint_index, _ in enumerate(grasping_points[str(label)]["hovering_joints_position"]):
                joint_goal[joint_index] = grasping_points[str(label)]["hovering_joints_position"][joint_index]
            group.go(joint_goal, wait=True)
            group.stop()

            joint_goal = group.get_current_joint_values()
            for joint_index, _ in enumerate(grasping_points[str(label)]["grasping_joints_position"]):
                joint_goal[joint_index] = grasping_points[str(label)]["grasping_joints_position"][joint_index]
            group.go(joint_goal, wait=True)
            group.stop()

            rospy.rostime.wallsleep(1)
                
            # raw_input("Press enter to grasp the stone.")
            grasp_message = GraspActionGoal()
            grasp_message.goal.width = 0.02
            grasp_message.goal.epsilon.inner = 0.01
            grasp_message.goal.epsilon.outer = 0.01
            grasp_message.goal.speed = 0.05
            grasp_message.goal.force = 0.01
            grasp_pub.publish(grasp_message)

            rospy.rostime.wallsleep(1.5)

            joint_goal = group.get_current_joint_values()
            for joint_index, _ in enumerate(grasping_points[str(label)]["hovering_joints_position"]):
                joint_goal[joint_index] = grasping_points[str(label)]["hovering_joints_position"][joint_index]
            group.go(joint_goal, wait=True)
            group.stop()

            pose_goal = geometry_msgs.msg.Pose()
            waypoints = []

            # raw_input("Press enter to start sensing.")
            sensors_data["raw"]["force"]["x"].append([])
            sensors_data["raw"]["force"]["y"].append([])
            sensors_data["raw"]["force"]["z"].append([])
            sensors_data["raw"]["torque"]["x"].append([])
            sensors_data["raw"]["torque"]["y"].append([])
            sensors_data["raw"]["torque"]["z"].append([])
            
            for pos in sensing_waypoints["joints_positions"]:
                sensing_step = {
                    "force": {
                        "x": [],
                        "y": [],
                        "z": []
                    },
                    "torque": {
                        "x": [],
                        "y": [],
                        "z": []
                    },
                }
                joint_goal = group.get_current_joint_values()
                for joint_index, _ in enumerate(pos):
                    joint_goal[joint_index] = pos[joint_index]
                group.go(joint_goal, wait=True)
                group.stop()
                sub = rospy.Subscriber("/panda/franka_state_controller/F_ext", geometry_msgs.msg.WrenchStamped, callback, [sensing_step])
                rospy.rostime.wallsleep(1)
                sub.unregister()
                
                sensors_data["raw"]["force"]["x"][-1].append(sensing_step["force"]["x"])
                sensors_data["raw"]["force"]["y"][-1].append(sensing_step["force"]["y"])
                sensors_data["raw"]["force"]["z"][-1].append(sensing_step["force"]["z"])
                sensors_data["raw"]["torque"]["x"][-1].append(sensing_step["torque"]["x"])
                sensors_data["raw"]["torque"]["y"][-1].append(sensing_step["torque"]["y"])
                sensors_data["raw"]["torque"]["z"][-1].append(sensing_step["torque"]["z"])

            joint_goal = group.get_current_joint_values()
            for joint_index, _ in enumerate(grasping_points[str(label)]["hovering_joints_position"]):
                joint_goal[joint_index] = grasping_points[str(label)]["hovering_joints_position"][joint_index]
            group.go(joint_goal, wait=True)
            group.stop()

            joint_goal = group.get_current_joint_values()
            for joint_index, _ in enumerate(grasping_points[str(label)]["grasping_joints_position"]):
                joint_goal[joint_index] = grasping_points[str(label)]["grasping_joints_position"][joint_index]
            group.go(joint_goal, wait=True)
            group.stop()

            gripper_move_message = MoveActionGoal()
            gripper_move_message.goal.width = 0.06
            gripper_move_message.goal.speed = 0.05
            gripper_move_pub.publish(gripper_move_message)

            rospy.rostime.wallsleep(1.5)

            joint_goal = group.get_current_joint_values()
            for joint_index, _ in enumerate(grasping_points[str(label)]["hovering_joints_position"]):
                joint_goal[joint_index] = grasping_points[str(label)]["hovering_joints_position"][joint_index]
            group.go(joint_goal, wait=True)
            group.stop()

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
            alpha = 0.99
            sensors_data["filtered"]["alpha"] = alpha
            sensors_data["raw"]["label"].append(label)
            sensors_data["filtered"]["subsamples"]["label"].append(label)
            
            # sensors_data["filtered"]["force"]["x"].append([item for sublist in sensors_data["raw"]["force"]["x"][-1] for item in filter_list(sublist, alpha)])
            # sensors_data["filtered"]["force"]["y"].append([item for sublist in sensors_data["raw"]["force"]["y"][-1] for item in filter_list(sublist, alpha)])
            # sensors_data["filtered"]["force"]["z"].append([item for sublist in sensors_data["raw"]["force"]["z"][-1] for item in filter_list(sublist, alpha)])
            # sensors_data["filtered"]["torque"]["x"].append([item for sublist in sensors_data["raw"]["torque"]["x"][-1] for item in filter_list(sublist, alpha)])
            # sensors_data["filtered"]["torque"]["y"].append([item for sublist in sensors_data["raw"]["torque"]["y"][-1] for item in filter_list(sublist, alpha)])
            # sensors_data["filtered"]["torque"]["z"].append([item for sublist in sensors_data["raw"]["torque"]["z"][-1] for item in filter_list(sublist, alpha)])
            
            sensors_data["filtered"]["force"]["x"] = [[item for sublist in sensors_data["raw"]["force"]["x"][-1] for item in filter_list(sublist, alpha)]]
            sensors_data["filtered"]["force"]["y"] = [[item for sublist in sensors_data["raw"]["force"]["y"][-1] for item in filter_list(sublist, alpha)]]
            sensors_data["filtered"]["force"]["z"] = [[item for sublist in sensors_data["raw"]["force"]["z"][-1] for item in filter_list(sublist, alpha)]]
            sensors_data["filtered"]["torque"]["x"] = [[item for sublist in sensors_data["raw"]["torque"]["x"][-1] for item in filter_list(sublist, alpha)]]
            sensors_data["filtered"]["torque"]["y"] = [[item for sublist in sensors_data["raw"]["torque"]["y"][-1] for item in filter_list(sublist, alpha)]]
            sensors_data["filtered"]["torque"]["z"] = [[item for sublist in sensors_data["raw"]["torque"]["z"][-1] for item in filter_list(sublist, alpha)]]

            sensors_data["filtered"]["subsamples"]["force"]["x"].append(subsample_list(sensors_data["filtered"]["force"]["x"][-1], 100))
            sensors_data["filtered"]["subsamples"]["force"]["y"].append(subsample_list(sensors_data["filtered"]["force"]["y"][-1], 100))
            sensors_data["filtered"]["subsamples"]["force"]["z"].append(subsample_list(sensors_data["filtered"]["force"]["z"][-1], 100))
            sensors_data["filtered"]["subsamples"]["torque"]["x"].append(subsample_list(sensors_data["filtered"]["torque"]["x"][-1], 100))
            sensors_data["filtered"]["subsamples"]["torque"]["y"].append(subsample_list(sensors_data["filtered"]["torque"]["y"][-1], 100))
            sensors_data["filtered"]["subsamples"]["torque"]["z"].append(subsample_list(sensors_data["filtered"]["torque"]["z"][-1], 100))

        # os.makedirs(output_folder)
        # os.makedirs(output_folder, exist_ok=True)
        with open(output_folder + "/raw_stone_dataset.txt", "w") as f:
            json.dump(sensors_data["raw"], f)
        with open(output_folder + "/stone_dataset.txt", "w") as f:
            json.dump(sensors_data["filtered"]["subsamples"], f)

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
    gripper_move_message.goal.width = 0.06
    gripper_move_message.goal.speed = 0.05
    gripper_move_pub.publish(gripper_move_message)

    print("End of sensing")

if __name__ == '__main__':
    try:
        input_folder = "latest"
        is_simulation = False
        tot_time_nsecs = 20000000000  # total execution time for the trajectory in nanoseconds

        print("input_folder = " + str(input_folder))
        print("tot_time_nsecs = " + str(tot_time_nsecs))

        sense_stone(labels=args.stones_labels)
    except rospy.ROSInterruptException:
        pass
