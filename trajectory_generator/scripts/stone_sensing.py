#!/usr/bin/env python
import json
import copy
import rospy
from franka_gripper.msg import GraspActionGoal
from franka_gripper.msg import MoveActionGoal
import geometry_msgs.msg
import moveit_msgs.msg
import sensor_msgs.msg


import ast
import sys
import moveit_commander
import getopt
import os
import thread

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

stop_flag = False
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
    }
}

effort_sensing_step = {
    "panda_joint1": [],
    "panda_joint2": [],
    "panda_joint3": [],
    "panda_joint4": [],
    "panda_joint5": [],
    "panda_joint6": [],
    "panda_joint7": []
}

def old_filtered_plot(l, labels):
    y = l[-1]
    sns.scatterplot(range(len(y)), y, color="purple", label=labels[-1], legend="full")
    y = l[-2]
    sns.scatterplot(range(len(y)), y, color="b", label=labels[-2], legend="full")
    y = l[-3]
    sns.scatterplot(range(len(y)), y, color="g", label=labels[-3], legend="full")
    y = l[-4]
    sns.scatterplot(range(len(y)), y, color="r", label=labels[-4], legend="full")
    plt.show()

def filtered_plot(l, labels, offset=0):
    fig = plt.figure("filtered_plot")
    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 2, 5)
    ax6 = fig.add_subplot(3, 2, 6)
    colors = ["r", "g", "b", "purple"]
    ax1.set_title("force")
    ax1.set_ylabel("x")
    ax2.set_title("torque")
    ax3.set_ylabel("y")
    ax5.set_ylabel("z")
    min_range = offset * len(args.stones_labels) + 1
    max_range = min_range + len(args.stones_labels)
    for i in range(min_range, max_range):
        y = l["force"]["x"][-i]
        g = sns.scatterplot(x=range(len(y)), y=y, color=colors[-i], label=labels[-i], legend="full", alpha=0.8, ax=ax1)
        y = l["force"]["y"][-i]
        g = sns.scatterplot(x=range(len(y)), y=y, color=colors[-i], label=labels[-i], legend="full", alpha=0.8, ax=ax3)
        y = l["force"]["z"][-i]
        g = sns.scatterplot(x=range(len(y)), y=y, color=colors[-i], label=labels[-i], legend="full", alpha=0.8, ax=ax5)
        y = l["torque"]["x"][-i]
        g = sns.scatterplot(x=range(len(y)), y=y, color=colors[-i], label=labels[-i], legend="full", alpha=0.8, ax=ax2)
        y = l["torque"]["y"][-i]
        g = sns.scatterplot(x=range(len(y)), y=y, color=colors[-i], label=labels[-i], legend="full", alpha=0.8, ax=ax4)
        y = l["torque"]["z"][-i]
        g = sns.scatterplot(x=range(len(y)), y=y, color=colors[-i], label=labels[-i], legend="full", alpha=0.8, ax=ax6)
    
    fig = plt.figure("filtered_effort_plot")
    colors = ["r", "g", "b", "purple"]
    min_range = offset * len(args.stones_labels) + 1
    max_range = min_range + len(args.stones_labels)
    for i in range(min_range, max_range):
        for joint_name in l["effort"]:
            ax = fig.add_subplot(4, 2, int(joint_name[-1]))
            ax.set_title(joint_name)
            y = l["effort"][joint_name][-i]
            g = sns.scatterplot(x=range(len(y)), y=y, color=colors[-i], label=labels[-i], legend="full", alpha=0.8, ax=ax)
    plt.show()

def old_raw_plot(l):
    y = [item for sublist in l[-1] for item in sublist]
    sns.scatterplot(range(len(y)), y, color="purple")
    y = [item for sublist in l[-2] for item in sublist]
    sns.scatterplot(range(len(y)), y, color="b")
    y = [item for sublist in l[-3] for item in sublist]
    sns.scatterplot(range(len(y)), y, color="g")
    y = [item for sublist in l[-4] for item in sublist]
    sns.scatterplot(range(len(y)), y, color="r")
    plt.show()

def raw_plot(l):
    fig = plt.figure("raw_plot")
    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 2, 5)
    ax6 = fig.add_subplot(3, 2, 6)
    colors = ["r", "g", "b", "purple"]
    ax1.set_title("force")
    ax1.set_ylabel("x")
    ax2.set_title("torque")
    ax3.set_ylabel("y")
    ax5.set_ylabel("z")
    for i in range(1, 5):
        y = [item for sublist in l["force"]["x"][-i] for item in sublist]
        g = sns.scatterplot(x=range(len(y)), y=y, color=colors[-i], legend="full", ax=ax1)
        y = [item for sublist in l["force"]["y"][-i] for item in sublist]
        g = sns.scatterplot(x=range(len(y)), y=y, color=colors[-i], legend="full", ax=ax3)
        y = [item for sublist in l["force"]["z"][-i] for item in sublist]
        g = sns.scatterplot(x=range(len(y)), y=y, color=colors[-i], legend="full", ax=ax5)
        y = [item for sublist in l["torque"]["x"][-i] for item in sublist]
        g = sns.scatterplot(x=range(len(y)), y=y, color=colors[-i], legend="full", ax=ax2)
        y = [item for sublist in l["torque"]["y"][-i] for item in sublist]
        g = sns.scatterplot(x=range(len(y)), y=y, color=colors[-i], legend="full", ax=ax4)
        y = [item for sublist in l["torque"]["z"][-i] for item in sublist]
        g = sns.scatterplot(x=range(len(y)), y=y, color=colors[-i], legend="full", ax=ax6)
    plt.show()

def pose_move(group=None, point=None):
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = point["position"][0]
    pose_goal.position.y = point["position"][1]
    pose_goal.position.z = point["position"][2]
    pose_goal.orientation.x = point["orientation"][0]
    pose_goal.orientation.y = point["orientation"][1]
    pose_goal.orientation.z = point["orientation"][2]
    pose_goal.orientation.w = point["orientation"][3]
    group.set_pose_target(pose_goal)
    plan = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

def joint_move(group=None, joint_coordinates=None):
    joint_goal = group.get_current_joint_values()
    for joint_index, _ in enumerate(joint_coordinates):
        joint_goal[joint_index] = joint_coordinates[joint_index]
    group.go(joint_goal, wait=True)
    group.stop()

def open_gripper(gripper_move_pub=None):
    gripper_move_message = MoveActionGoal()
    gripper_move_message.goal.width = 0.06
    gripper_move_message.goal.speed = 0.05
    gripper_move_pub.publish(gripper_move_message)

def grasp_stone(grasp_pub=None):
    grasp_message = GraspActionGoal()
    grasp_message.goal.width = 0.02
    grasp_message.goal.epsilon.inner = 0.01
    grasp_message.goal.epsilon.outer = 0.01
    grasp_message.goal.speed = 0.05
    grasp_message.goal.force = 0.01
    grasp_pub.publish(grasp_message)

def check_grasp(robot=None, gripper_move_pub=None):
    gripper_width = robot.get_current_state().joint_state.position[-1] + robot.get_current_state().joint_state.position[-2]
    if gripper_width < 0.028 or gripper_width > 0.030:
        return False
    else:
        return True

def subsample_list(list_to_subsample, samples_nr):
    if samples_nr > len(list_to_subsample):
        raw_input("The list does not contain enough samples: {} samples requested, but {} samples are in the list".format(samples_nr, len(list_to_subsample)))
    real_sample_interval = len(list_to_subsample)/float(samples_nr)
    current_sample_interval = int(real_sample_interval)
    sample_interval_avg = None
    sample_index = 0
    test = []
    subsamples = []
    while len(subsamples) < samples_nr:
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
    global sensing_step
    # sensing_step = args[0]
    sensing_step["force"]["x"].append(data.wrench.force.x)
    sensing_step["force"]["y"].append(data.wrench.force.y)
    sensing_step["force"]["z"].append(data.wrench.force.z)
    sensing_step["torque"]["x"].append(data.wrench.torque.x)
    sensing_step["torque"]["y"].append(data.wrench.torque.y)
    sensing_step["torque"]["z"].append(data.wrench.torque.z)

def callback_effort(data, args):
    global effort_sensing_step
    # sensing_step = args[0]
    for joint_index, joint_name in enumerate(data.name):
        effort_sensing_step[joint_name].append(data.effort[joint_index])

def listener():
    global stop_flag
    force_sub = rospy.Subscriber("/panda/franka_state_controller/F_ext", geometry_msgs.msg.WrenchStamped, callback, [])
    effort_sub = rospy.Subscriber("/panda/franka_state_controller/joint_states", sensor_msgs.msg.JointState, callback_effort, [])
    while True:
        rospy.rostime.wallsleep(0.3)
        if stop_flag == True:
            force_sub.unregister()
            effort_sub.unregister()
            break
    # rospy.spin()

def sense_stone(output_folder="latest", is_simulation=False, is_learning=True, repetitions=100, labels=None):
    global sensing_step
    global effort_sensing_step
    global stop_flag
    if is_learning:
        is_simulation = False
        repetitions = 1
        labels = ["learning_stone"]
    package_path = rospack.get_path("trajectory_generator")
    if not is_learning:
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
            "effort": {
                "panda_joint1": [],
                "panda_joint2": [],
                "panda_joint3": [],
                "panda_joint4": [],
                "panda_joint5": [],
                "panda_joint6": [],
                "panda_joint7": []
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
            "effort": {
                "panda_joint1": [],
                "panda_joint2": [],
                "panda_joint3": [],
                "panda_joint4": [],
                "panda_joint5": [],
                "panda_joint6": [],
                "panda_joint7": []
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
                "effort": {
                    "panda_joint1": [],
                    "panda_joint2": [],
                    "panda_joint3": [],
                    "panda_joint4": [],
                    "panda_joint5": [],
                    "panda_joint6": [],
                    "panda_joint7": []
                },
                "label": []
            }
        }
    }
    if not is_learning:
        rospy.init_node('force_reader', anonymous=True)
    rate = rospy.Rate(0.1)  # hz

    if is_simulation:
        print("SIMULATION MODE")
        # robot_desctiption_str = "robot_description"
        # namespace = ""
        grasp_pub = rospy.Publisher('/franka_gripper/grasp/goal',
                            GraspActionGoal, queue_size=10)
        gripper_move_pub = rospy.Publisher('/franka_gripper/move/goal',
                            MoveActionGoal, queue_size=10)
    else:
        robot_desctiption_str = "/panda/robot_description"
        namespace = "panda"
        grasp_pub = rospy.Publisher('/panda/franka_gripper/grasp/goal',
                            GraspActionGoal, queue_size=10)
        gripper_move_pub = rospy.Publisher('/panda/franka_gripper/move/goal',
                            MoveActionGoal, queue_size=10)

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
            "position": [0.360, 0, 0.959], # x, y, z
            "hovering_joints_position": [0.5142469373983083, 0.58864791404992, 0.4109301574533518, -1.7464407590397617, -0.2942778061838653, 2.2733246678428056, 0.26931423637248975],
            "grasping_joints_position": [0.5584349823003857, 0.7669955060133635, 0.34803027073230886, -1.7289693730492495, -0.372046103130778, 2.4307231125368634, 0.3161871728630494]
        },
        "20": {
            "orientation": [-0.9239, 0.3826, -7.02665323908824e-05, 0.0001], # x, y, z, w
            "position": [0.210, 0, 0.959], # x, y, z
            "hovering_joints_position": [0.41443757597396247, 0.46631249645310835, 0.4235820091639901, -1.9450973129794462, -0.26485765947898227, 2.35714606514291, 0.2023833901636485],
            "grasping_joints_position": [0.46923908052318963, 0.6613585829651146, 0.3477711293421327, -1.9254993174133115, -0.37074258951259165, 2.5235585845190327, 0.2691860056346718]
        },
        "30": {
            "orientation": [-0.9239, 0.3826, -7.02665323908824e-05, 0.0001], # x, y, z, w
            "position": [0.060, 0, 0.959], # x, y, z
            "hovering_joints_position": [0.29283302700937835, 0.3594985494195369, 0.4567617479473339, -2.113724579560129, -0.23796098346843167, 2.4241980358757487, 0.12007181211959518],
            "grasping_joints_position": [0.36017622283046086, 0.5738621895689713, 0.36335281852253676, -2.0926987548863014, -0.386788282930851, 2.6037577000270917, 0.21925989602626683]
        },
        "40": {
            "orientation": [-0.9239, 0.3826, -7.02665323908824e-05, 0.0001], # x, y, z, w
            "position": [-0.090, 0, 0.959], # x, y, z
            "hovering_joints_position": [0.23743921274672905, 0.2517310960090678, 0.419285442544725, -2.2607955156292827, -0.16672964333228762, 2.485045480199256, -0.007544105715858234],
            "grasping_joints_position": [0.3001652932277772, 0.4882139589084475, 0.3272152716010523, -2.2381679057515567, -0.34320602850340026, 2.679322558014489, 0.11473005577259905]
        },
        "learning_stone": {
            "orientation": [-0.9239, 0.3826, -7.02665323908824e-05, 0.0001], # x, y, z, w
            "position": [0.360, 0, 0.959], # x, y, z
            "hovering_joints_position": [-0.22418082891116864, -0.021118342585632, 0.01563296593681296, -2.6142579815562224, -0.00010159493463057645, 2.5934383369042777, -0.9943609043742773],
            "grasping_joints_position": [-0.22022550052894801, 0.27592300802189773, 0.009137543043775277, -2.598599127845808, -0.01297810329839507, 2.8747516451494377, -0.9849187362178262]
        }
    }

    grasping_point = {
        "orientation": [-0.9239, 0.3826, -7.02665323908824e-05, 0.0001], # x, y, z, w
        "position": [0.110, 0.063, 0.859], # x, y, z
        "joints_position": [0.23256071986483895, 0.1849605811604282, 0.05070528308319614, -2.772702876833163, -0.04564475351126129, 2.956572071540546, -0.45859744564697985]
    }
    sliding_point = {
        "orientation": [-0.9239, 0.3826, -7.02665323908824e-05, 0.0001], # x, y, z, w
        "position": [-0.462, -0.130, 0.869], # x, y, z
        "joints_position": [-0.3774935741110852, 0.9350374934419975, -0.410131176740132, -1.422043059332329, 0.430936304192417, 2.2673295368043322, -1.7027889850026334]
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

            # almost max y
            # [0.661, -0.251, 0.300, 0.639],
        ],
        # x, y, z
        "positions": [
            [0.110, -0.025, 1.100],

            [0.125, 0.025, 1.120],
            [0.145, -0.025, 1.130],
            [0.125, -0.075, 1.120],
            [0.095, 0.025, 1.120],
            [0.075, -0.055, 1.130],
            [0.095, -0.075, 1.120],

            [0.110, -0.025, 1.100],

            # almost max y
            # [0.110, -0.025, 1.331],
        ],
        "predefined_joints_path": [
            [0.5061984524379249, -0.17315264765564847, -0.22725228734664832, -2.7905919022154224, -0.07659106785917477, 2.618745111407918, -0.4369418799075402],
            [0.22389371792057103, -0.4881079590929628, 0.06878514456188034, -2.6440227623989645, 0.038908337513605784, 2.1557850458984777, -0.522047535885542]
        ],
        "joints_pos_not_to_record": [
            # normal
            [0.19445120013816, -0.21487730565614868, 0.04427450973878616, -2.3955809373590724, 0.01187810182718171, 2.180555477353596, -0.5547206810216108],
            # # almost max force on y 2.0
            # [0.21064606387971807, -0.6230041403551178, 0.04318502222761333, -2.5824178610349957, -0.06834458731382768, 3.008495783487955, -0.5523243208382189],
            # # # normal
            # # [0.19445120013816, -0.21487730565614868, 0.04427450973878616, -2.3955809373590724, 0.01187810182718171, 2.180555477353596, -0.5547206810216108],
            # # almost max force on x
            # [0.3577381551594371, -0.2212658614213006, -0.15405057695574867, -2.350672152502495, 1.4752342208280778, 1.8857782667261198, -1.3482390514999687],
            # # normal
            # [0.19445120013816, -0.21487730565614868, 0.04427450973878616, -2.3955809373590724, 0.01187810182718171, 2.180555477353596, -0.5547206810216108],
        ],
        "joints_positions": [
            # {
            #     # joints rotation initial
            #     "pos": [0.09501817749126393, -0.9102240341824137, -0.06039151950529718, -2.498833895985301, -0.048188401254747396, 1.587717976456009, -0.7283916825025041],
            #     "record": False
            # },
            {
                # joint 5 rotation right
                "pos": [0.09501817749126393, -0.9102240341824137, -0.06039151950529718, -2.498833895985301, -1.207188401254747396, 1.587717976456009, -0.7283916825025041],
                "record": "start"
            },
            {
                # joint 5 rotation left
                "pos": [0.09501817749126393, -0.9102240341824137, -0.06039151950529718, -2.498833895985301, 1.207188401254747396, 1.587717976456009, -0.7283916825025041],
                "record": "continue"
            },
            {
                # joint 5 rotation right
                "pos": [0.09501817749126393, -0.9102240341824137, -0.06039151950529718, -2.498833895985301, -1.207188401254747396, 1.587717976456009, -0.7283916825025041],
                "record": "end"
            },
            # {
            #     # joint 5 rotation left
            #     "pos": [0.09501817749126393, -0.9102240341824137, -0.06039151950529718, -2.498833895985301, 1.207188401254747396, 1.587717976456009, -0.7283916825025041],
            #     "record": "continue"
            # },
            # {
            #     # joint 5 rotation right
            #     "pos": [0.09501817749126393, -0.9102240341824137, -0.06039151950529718, -2.498833895985301, -1.207188401254747396, 1.587717976456009, -0.7283916825025041],
            #     "record": "end"
            # },
            # {
            #     # lift start
            #     "pos": [0.6674143791419309, 0.7977373682437291, 0.061897229766636565, -1.422579090469762, -0.05602355088606354, 2.21777426317003, -0.04117076196962211],
            #     "record": "start"
            # },
            # {
            #     # lift end
            #     "pos": [0.6674143791419309, 0.2077373682437291, 0.061897229766636565, -1.422579090469762, -0.05602355088606354, 2.21777426317003, -0.04117076196962211],
            #     "record": "continue"
            # },
            # {
            #     # lift start
            #     "pos": [0.6674143791419309, 0.7977373682437291, 0.061897229766636565, -1.422579090469762, -0.05602355088606354, 2.21777426317003, -0.04117076196962211],
            #     "record": "continue"
            # },
            # {
            #     # lift end
            #     "pos": [0.6674143791419309, 0.2077373682437291, 0.061897229766636565, -1.422579090469762, -0.05602355088606354, 2.21777426317003, -0.04117076196962211],
            #     "record": "continue"
            # },
            # {
            #     # lift start
            #     "pos": [0.6674143791419309, 0.7977373682437291, 0.061897229766636565, -1.422579090469762, -0.05602355088606354, 2.21777426317003, -0.04117076196962211],
            #     "record": "end"
            # },
            # {
            #     # joints rotation initial
            #     "pos": [0.09501817749126393, -0.9102240341824137, -0.06039151950529718, -2.498833895985301, -0.048188401254747396, 1.587717976456009, -0.7283916825025041],
            #     "record": "continue"
            # },
            # {
            #     # joint 6 rotation front
            #     "pos": [0.09501817749126393, -0.9102240341824137, -0.06039151950529718, -2.498833895985301, -0.048188401254747396, 3.007717976456009, -0.7283916825025041],
            #     "record": "continue"
            # },
            # {
            #     # joints rotation initial
            #     "pos": [0.09501817749126393, -0.9102240341824137, -0.06039151950529718, -2.498833895985301, -0.048188401254747396, 1.587717976456009, -0.7283916825025041],
            #     "record": "end"
            # },
            # {
            #     # normal
            #     "pos": [0.19445120013816, -0.21487730565614868, 0.04427450973878616, -2.3955809373590724, 0.01187810182718171, 2.180555477353596, -0.5547206810216108],
            #     "record": False
            # },
            # {
            #     # higher normal
            #     "pos": [0.1945937183465583, -0.29763553907470264, 0.04148949768466628, -2.216595536661565, 0.012573500959566938, 1.9181392465227391, -0.5571296332728874],
            #     "record": False
            # },
            # {
            #     # normal
            #     "pos": [0.19445120013816, -0.21487730565614868, 0.04427450973878616, -2.3955809373590724, 0.01187810182718171, 2.180555477353596, -0.5547206810216108],
            #     "record": True
            # },
            # {
            #     # almost max force on x
            #     "pos": [0.3577381551594371, -0.2212658614213006, -0.15405057695574867, -2.350672152502495, 1.4752342208280778, 1.8857782667261198, -1.3482390514999687],
            #     "record": False
            # },
            # {
            #     # higher almost max force on x
            #     "pos": [0.3568993793187499, -0.30982427137776425, -0.14553087910982557, -2.2229064250856636, 1.4001052575314026, 1.8717732232276834, -1.1211786885173198],
            #     "record": False
            # },
            # {
            #     # almost max force on x
            #     "pos": [0.3577381551594371, -0.2212658614213006, -0.15405057695574867, -2.350672152502495, 1.4752342208280778, 1.8857782667261198, -1.3482390514999687],
            #     "record": True
            # }

    #         # 8 shape
    # #         [0.3351801368290918, -0.29662777607797913, -0.11244693900398067, -2.3844817042769044, 0.4160744979646716, 1.7924010314720245, -1.3208833060871592],
    #         [0.3619200299155657, -0.23360565089133745, -0.12781654499551712, -2.3745498456453022, 0.978863298841051, 2.1175465971498175, -1.117479492057386],
    # # # [0.30448947731444703, -0.12694879574524728, -0.051233081743954974, -2.3243839509474378, 0.8545456072769047, 2.48963434716984, -0.5527099020836809],
    # #         [0.15692759708981763, -0.2920189494293214, 0.12201354729948073, -2.3580899385820357, -0.21012455755472154, 1.705037037051065, 0.0049194150333632955],
    #         [0.12587268760957215, -0.11777158826903292, 0.16212125076698525, -2.21868566964802, -0.8504195135672374, 1.77099568547142, -0.2145776784027563],
    # # # [0.14789706793375182, -0.13316567956848385, 0.0641090490445283, -2.301253765842371, -0.7098326927480935, 2.3117149482243375, -0.8717583615683947],

            # # max force on x
            # [0.3580385840838415, -0.22323768461422525, -0.1588374459835385, -2.355844996096438, 1.6573459696700605, 1.7795040021472508, -1.349393375527316],
            # # almost max force on y
            # [0.23969123298243467, -0.1792329619555566, -0.12297258245354055, -2.256062999568857, 0.08960046780666968, 3.6497432763667486, -0.8688858720449072],
            # # almost max force on y 2.0
            # [0.21064606387971807, -0.6230041403551178, 0.04318502222761333, -2.5824178610349957, -0.06834458731382768, 3.008495783487955, -0.5523243208382189],

            # # almost max force on x
            # [0.3577381551594371, -0.2212658614213006, -0.15405057695574867, -2.350672152502495, 1.4752342208280778, 1.8857782667261198, -1.3482390514999687],
            # # normal
            # [0.19445120013816, -0.21487730565614868, 0.04427450973878616, -2.3955809373590724, 0.01187810182718171, 2.180555477353596, -0.5547206810216108],
        ]
    }
    pos_not_to_record = len(sensing_waypoints["joints_pos_not_to_record"])

    rospy.rostime.wallsleep(1)
    open_gripper(gripper_move_pub)
    rospy.rostime.wallsleep(1)

    for repetition in range(repetitions):
        print ("\nRepetition " + str(repetition+1) + "/" + str(repetitions))
        # if type(labels) == list and len(labels) > 1:
        #     random.shuffle(labels)
        
        for label_index, label in enumerate(labels):
            print ("Now sensing stone with label '" + str(label) + "'...")
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
            if label_index == 0:
                joint_move(group, grasping_points[str(label)]["hovering_joints_position"])
            joint_move(group, grasping_points[str(label)]["grasping_joints_position"])

            rospy.rostime.wallsleep(0.5)
            
            stone_grasped = False
            while not stone_grasped:
                # raw_input("Press enter to grasp the stone.")
                grasp_stone(grasp_pub)
                rospy.rostime.wallsleep(1.5)
                stone_grasped = check_grasp(robot, gripper_move_pub)
                if not stone_grasped:
                    open_gripper(gripper_move_pub)
                    raw_input("Grasping was unsuccesfull, reposition the stone and press enter.\n")


            joint_move(group, grasping_points[str(label)]["hovering_joints_position"])
            # pose_goal = geometry_msgs.msg.Pose()
            waypoints = []

            # raw_input("Press enter to start sensing.")
            sensors_data["raw"]["force"]["x"].append([])
            sensors_data["raw"]["force"]["y"].append([])
            sensors_data["raw"]["force"]["z"].append([])
            sensors_data["raw"]["torque"]["x"].append([])
            sensors_data["raw"]["torque"]["y"].append([])
            sensors_data["raw"]["torque"]["z"].append([])
            for joint_name in effort_sensing_step:
                sensors_data["raw"]["effort"][joint_name].append([])
            
            stone_grasped = False
            while not stone_grasped:
                if (not is_learning) and label != "learning_stone":
                    joint_move(group, grasping_points["learning_stone"]["grasping_joints_position"])
                    joint_move(group, grasping_points["learning_stone"]["hovering_joints_position"])
                # for pos in sensing_waypoints["predefined_joints_path"]:
                #     joint_move(group, pos)
                # for pos in sensing_waypoints["joints_pos_not_to_record"]:
                #     joint_move(group, pos)
                for item in sensing_waypoints["joints_positions"]:
                    if item["record"] in [True, "start"]:
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
                        for joint_name in effort_sensing_step:
                            effort_sensing_step[joint_name] = []
                    joint_move(group, item["pos"])
                    if item["record"] == "start":
                        stop_flag = False
                        # group.set_max_velocity_scaling_factor(0.05)
                        # group.set_max_acceleration_scaling_factor(0.1)
                        group.set_max_velocity_scaling_factor(0.2)
                        group.set_max_acceleration_scaling_factor(0.3)
                        thread.start_new_thread(listener, ())
                        rospy.rostime.wallsleep(1)
                    if item["record"] == True:
                        rospy.rostime.wallsleep(1)
                        force_sub = rospy.Subscriber("/panda/franka_state_controller/F_ext", geometry_msgs.msg.WrenchStamped, callback, [])
                        effort_sub = rospy.Subscriber("/panda/franka_state_controller/joint_states", sensor_msgs.msg.JointState, callback_effort, [])
                        rospy.rostime.wallsleep(4)
                        force_sub.unregister()
                        effort_sub.unregister()
                    elif item["record"] == "end":
                        rospy.rostime.wallsleep(0.5)
                        stop_flag = True
                        group.set_max_velocity_scaling_factor(1)
                        group.set_max_acceleration_scaling_factor(1)
                        rospy.rostime.wallsleep(0.5)
                    


                    stone_grasped = check_grasp(robot, gripper_move_pub)
                    if not stone_grasped:
                        if item["record"] != False:
                            sensors_data["raw"]["force"]["x"][-1] = []
                            sensors_data["raw"]["force"]["y"][-1] = []
                            sensors_data["raw"]["force"]["z"][-1] = []
                            sensors_data["raw"]["torque"]["x"][-1] = []
                            sensors_data["raw"]["torque"]["y"][-1] = []
                            sensors_data["raw"]["torque"]["z"][-1] = []
                            for joint_name in effort_sensing_step:
                                sensors_data["raw"]["effort"][joint_name][-1] = []
                        print("Stone fell off while sensing, repositioning gripper...")
                        rospy.rostime.wallsleep(0.5)
                        joint_move(group, grasping_points[str(label)]["hovering_joints_position"])
                        joint_move(group, grasping_points[str(label)]["grasping_joints_position"])
                        open_gripper(gripper_move_pub)
                        rospy.rostime.wallsleep(0.5)
                        raw_input("Reposition the stone and press enter to resume sensing.\n")
                        while not stone_grasped:
                            grasp_stone(grasp_pub)
                            rospy.rostime.wallsleep(1.5)
                            stone_grasped = check_grasp(robot, gripper_move_pub)
                            if not stone_grasped:
                                rospy.rostime.wallsleep(0.5)
                                open_gripper(gripper_move_pub)
                                raw_input("Grasping was unsuccesfull, reposition the stone and press enter.\n")
                                rospy.rostime.wallsleep(0.5)
                        stone_grasped = False
                        break
                
                    if item["record"] in [True, "end"]:
                        sensors_data["raw"]["force"]["x"][-1].append(sensing_step["force"]["x"])
                        sensors_data["raw"]["force"]["y"][-1].append(sensing_step["force"]["y"])
                        sensors_data["raw"]["force"]["z"][-1].append(sensing_step["force"]["z"])
                        sensors_data["raw"]["torque"]["x"][-1].append(sensing_step["torque"]["x"])
                        sensors_data["raw"]["torque"]["y"][-1].append(sensing_step["torque"]["y"])
                        sensors_data["raw"]["torque"]["z"][-1].append(sensing_step["torque"]["z"])
                        for joint_name in effort_sensing_step:
                            sensors_data["raw"]["effort"][joint_name][-1].append(effort_sensing_step[joint_name])

            if not is_learning:
                joint_move(group, grasping_points[str(label)]["hovering_joints_position"])
                joint_move(group, grasping_points[str(label)]["grasping_joints_position"])

                open_gripper(gripper_move_pub)

                rospy.rostime.wallsleep(1)

                if (label_index+1) == len(labels):
                    joint_move(group, grasping_points[str(label)]["hovering_joints_position"])

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
            number_of_readings = len([item for item in sensing_waypoints["joints_positions"] if item["record"] == True]) -1
            # number_of_readings = len(sensing_waypoints["joints_positions"]) -1
            # number_of_readings = len(sensing_waypoints["joints_positions"]) - pos_not_to_record -1
            # min_sublist_len = 1
            min_sublist_len = int(100/number_of_readings) + 1

            alpha = 0.9
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
            for joint_name in effort_sensing_step:
                sensors_data["filtered"]["effort"][joint_name] = [[item for sublist in sensors_data["raw"]["effort"][joint_name][-1] for item in filter_list(sublist, alpha)]]
            
            # sensors_data["filtered"]["force"]["x"] = [[item for sublist in sensors_data["raw"]["force"]["x"][-1] for item in [sum(sublist)/len(sublist)]*len(sublist)]]
            # sensors_data["filtered"]["force"]["y"] = [[item for sublist in sensors_data["raw"]["force"]["y"][-1] for item in [sum(sublist)/len(sublist)]*len(sublist)]]
            # sensors_data["filtered"]["force"]["z"] = [[item for sublist in sensors_data["raw"]["force"]["z"][-1] for item in [sum(sublist)/len(sublist)]*len(sublist)]]
            # sensors_data["filtered"]["torque"]["x"] = [[item for sublist in sensors_data["raw"]["torque"]["x"][-1] for item in [sum(sublist)/len(sublist)]*len(sublist)]]
            # sensors_data["filtered"]["torque"]["y"] = [[item for sublist in sensors_data["raw"]["torque"]["y"][-1] for item in [sum(sublist)/len(sublist)]*len(sublist)]]
            # sensors_data["filtered"]["torque"]["z"] = [[item for sublist in sensors_data["raw"]["torque"]["z"][-1] for item in [sum(sublist)/len(sublist)]*len(sublist)]]

            # sensors_data["filtered"]["force"]["x"] = [[item for sublist in sensors_data["raw"]["force"]["x"][-1] for item in [sum(sublist)/len(sublist)]*min_sublist_len]]
            # sensors_data["filtered"]["force"]["y"] = [[item for sublist in sensors_data["raw"]["force"]["y"][-1] for item in [sum(sublist)/len(sublist)]*min_sublist_len]]
            # sensors_data["filtered"]["force"]["z"] = [[item for sublist in sensors_data["raw"]["force"]["z"][-1] for item in [sum(sublist)/len(sublist)]*min_sublist_len]]
            # sensors_data["filtered"]["torque"]["x"] = [[item for sublist in sensors_data["raw"]["torque"]["x"][-1] for item in [sum(sublist)/len(sublist)]*min_sublist_len]]
            # sensors_data["filtered"]["torque"]["y"] = [[item for sublist in sensors_data["raw"]["torque"]["y"][-1] for item in [sum(sublist)/len(sublist)]*min_sublist_len]]
            # sensors_data["filtered"]["torque"]["z"] = [[item for sublist in sensors_data["raw"]["torque"]["z"][-1] for item in [sum(sublist)/len(sublist)]*min_sublist_len]]
            # for joint_name in effort_sensing_step:
            #     sensors_data["filtered"]["subsamples"]["effort"][joint_name] = [[item for sublist in sensors_data["raw"]["effort"][joint_name][-1] for item in [sum(sublist)/len(sublist)]*min_sublist_len]]

            sensors_data["filtered"]["subsamples"]["force"]["x"].append(subsample_list(sensors_data["filtered"]["force"]["x"][-1], 100))
            sensors_data["filtered"]["subsamples"]["force"]["y"].append(subsample_list(sensors_data["filtered"]["force"]["y"][-1], 100))
            sensors_data["filtered"]["subsamples"]["force"]["z"].append(subsample_list(sensors_data["filtered"]["force"]["z"][-1], 100))
            sensors_data["filtered"]["subsamples"]["torque"]["x"].append(subsample_list(sensors_data["filtered"]["torque"]["x"][-1], 100))
            sensors_data["filtered"]["subsamples"]["torque"]["y"].append(subsample_list(sensors_data["filtered"]["torque"]["y"][-1], 100))
            sensors_data["filtered"]["subsamples"]["torque"]["z"].append(subsample_list(sensors_data["filtered"]["torque"]["z"][-1], 100))
            for joint_name in effort_sensing_step:
                sensors_data["filtered"]["subsamples"]["effort"][joint_name].append(subsample_list(sensors_data["filtered"]["effort"][joint_name][-1], 100))

            # sensors_data["filtered"]["subsamples"]["force"]["x"].append(subsample_list(sensors_data["filtered"]["force"]["x"][-1], len(sensors_data["filtered"]["force"]["x"][-1])))
            # sensors_data["filtered"]["subsamples"]["force"]["y"].append(subsample_list(sensors_data["filtered"]["force"]["y"][-1], len(sensors_data["filtered"]["force"]["y"][-1])))
            # sensors_data["filtered"]["subsamples"]["force"]["z"].append(subsample_list(sensors_data["filtered"]["force"]["z"][-1], len(sensors_data["filtered"]["force"]["z"][-1])))
            # sensors_data["filtered"]["subsamples"]["torque"]["x"].append(subsample_list(sensors_data["filtered"]["torque"]["x"][-1], len(sensors_data["filtered"]["torque"]["x"][-1])))
            # sensors_data["filtered"]["subsamples"]["torque"]["y"].append(subsample_list(sensors_data["filtered"]["torque"]["y"][-1], len(sensors_data["filtered"]["torque"]["y"][-1])))
            # sensors_data["filtered"]["subsamples"]["torque"]["z"].append(subsample_list(sensors_data["filtered"]["torque"]["z"][-1], len(sensors_data["filtered"]["torque"]["z"][-1])))

        if not is_learning:
            # os.makedirs(output_folder)
            # os.makedirs(output_folder, exist_ok=True)
            with open(output_folder + "/raw_stone_dataset.txt", "w") as f:
                json.dump(sensors_data["raw"], f)
            with open(output_folder + "/stone_dataset.txt", "w") as f:
                json.dump(sensors_data["filtered"]["subsamples"], f)

        if is_learning:
            joint_move(group, sliding_point["joints_position"])
            open_gripper(gripper_move_pub)
            rospy.rostime.wallsleep(1)
            return sensors_data["filtered"]["subsamples"]

        # if (repetition+1) % 10 == 0:
        #     print("Round {} finished".format(int(repetition/10)))


    print("End of sensing")

if __name__ == '__main__':
    try:
        input_folder = "latest"
        is_simulation = False
        tot_time_nsecs = 20000000000  # total execution time for the trajectory in nanoseconds

        print("input_folder = " + str(input_folder))
        print("tot_time_nsecs = " + str(tot_time_nsecs))

        # sense_stone(labels=["learning_stone"], repetitions=1, is_learning=True)
        sense_stone(labels=args.stones_labels, repetitions=args.repetitions, is_learning=False)
    except rospy.ROSInterruptException:
        pass
