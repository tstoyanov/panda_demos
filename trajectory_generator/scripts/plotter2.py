#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from rospy_message_converter import json_message_converter

import json, ast, collections, sys, getopt

scale = 1
folder_name = "/home/aass/workspace_shuffle/python_workspace/joints_check/logs/latest"
# if len(sys.arvg) == 2:
#     folder_name = sys.argv[1]

try:
    opts, args = getopt.getopt(sys.argv[1:],"i:s:",["input=","scale="])
except getopt.GetoptError:
    print("test.py -i <input_folder> -s <reduction_scale_for_efforts>")
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print("test.py -i <input_folder> -s <reduction_scale_for_efforts>")
        sys.exit()
    elif opt in ("-i", "--input"):
        folder_name = arg
    elif opt in ("-s", "--scale"):
        if arg > 0:
            scale = arg
        else:
            print("Scale factor must be greater than zero\n\tScale factor value: " + str(arg))
            sys.exit()

# with open("/home/ilbetzy/orebro/joints_check/logs/"+folder_name+"/json/position_joint_trajectory_controller-follow_joint_trajectory-feedback.mylog", 'r') as f:
with open(folder_name+"/json/panda-position_joint_trajectory_controller-follow_joint_trajectory-feedback.mylog", 'r') as f:
    data = f.read()
message_feedback = json.loads(data)
message_feedback = ast.literal_eval(json.dumps(message_feedback))
list_message_feedback = []
for message in message_feedback:
    dump = json.dumps(message)
    list_message_feedback.append(json_message_converter.convert_json_to_ros_message('control_msgs/FollowJointTrajectoryActionFeedback', dump))

# with open("/home/ilbetzy/orebro/joints_check/logs/"+folder_name+"/json/joint_states.mylog", 'r') as f:
with open(folder_name+"/json/panda-joint_states.mylog", 'r') as f:
    data = f.read()
message_joint_states = json.loads(data)
message_joint_states = ast.literal_eval(json.dumps(message_joint_states))
list_message_joint_states = []
for message in message_joint_states:
    dump = json.dumps(message)
    list_message_joint_states.append(json_message_converter.convert_json_to_ros_message('sensor_msgs/JointState', dump))

with open(folder_name+"/json/panda-position_joint_trajectory_controller-follow_joint_trajectory-goal.mylog", 'r') as f:
    data = f.read()
message_goal = json.loads(data)
message_goal = ast.literal_eval(json.dumps(message_goal))
list_message_goal = []
for message in message_goal:
    dump = json.dumps(message)
    list_message_goal.append(json_message_converter.convert_json_to_ros_message('control_msgs/FollowJointTrajectoryActionGoal', dump))


# =============================================================================
# feedback processing
joint_names_feedback = list_message_feedback[0].feedback.joint_names
feedback = {
    "desired": {
        "positions": {},
        "velocities": {},
        "accelerations": {}
    },
    "actual": {
        "positions": {},
        "velocities": {},
        "accelerations": {}
    },
    "error": {
        "positions": {},
        "velocities": {},
        "accelerations": {}
    }
}
for t in feedback:
    for dimension in feedback[t]:
        for i in range(len(joint_names_feedback)):
            feedback[t][dimension][joint_names_feedback[i]] = []
ts_feedback = []

for message in list_message_feedback:
    ts_feedback.append((message.header.stamp.secs * 1000000000) + message.header.stamp.nsecs)
    for t in feedback:
        tt = getattr(message.feedback, t)
        for dimension in feedback[t]:
            d = getattr(tt, dimension)
            for i in range(len(d)):
                feedback[t][dimension][joint_names_feedback[i]].append(d[i])

# =============================================================================
# joint_states processing
joint_names_joint_states = list_message_joint_states[0].name
positions_joint_states = {}
velocities_joint_states = {}
efforts_joint_states = {}
for i in range(len(joint_names_joint_states)):
    positions_joint_states[joint_names_joint_states[i]] = []
    velocities_joint_states[joint_names_joint_states[i]] = []
    efforts_joint_states[joint_names_joint_states[i]] = []
ts_joint_states = []

for message in list_message_joint_states:
    ts_joint_states.append((message.header.stamp.secs * 1000000000) + message.header.stamp.nsecs)
    for i in range(len(joint_names_joint_states)):
        positions_joint_states[joint_names_joint_states[i]].append(message.position[i])
        velocities_joint_states[joint_names_joint_states[i]].append(message.velocity[i])
        # efforts_joint_states[joint_names_joint_states[i]].append(message.effort[i])
        efforts_joint_states[joint_names_joint_states[i]].append(message.effort[i] / float(scale))

# =============================================================================
# goal processing
joint_names_goal = list_message_goal[0].goal.trajectory.joint_names
positions_goal = {}
velocities_goal = {}
accelerations_goal = {}

for i in range(len(joint_names_goal)):
    positions_goal[joint_names_goal[i]] = []
    velocities_goal[joint_names_goal[i]] = []
    accelerations_goal[joint_names_goal[i]] = []
ts_goal = []

for goal in list_message_goal:
    for point in goal.goal.trajectory.points:
        ts_goal.append((goal.header.stamp.secs * 1000000000) + goal.header.stamp.nsecs + (point.time_from_start.secs * 1000000000) + point.time_from_start.nsecs)
        for i in range(len(joint_names_goal)):
            positions_goal[joint_names_goal[i]].append(point.positions[i])
            velocities_goal[joint_names_goal[i]].append(point.velocities[i])
            # accelerations_goal[joint_names_goal[i]].append(point.accelerations[i])
            if (len(point.accelerations) != 0):
                accelerations_goal[joint_names_goal[i]].append(point.accelerations[i] / float(scale))
            else:
                accelerations_goal[joint_names_goal[i]].append([0] * len(joint_names_goal))

# =============================================================================
# computing diffs
positions_diffs = {}
velocities_diffs = {}
# efforts_diffs = {}
# positions_goal_interp = {}
# velocities_goal_interp = {}
positions_feedback_interp = {}
velocities_feedback_interp = {}
# accelerations_goal_interp = {}
for joint_name in joint_names_feedback:
    # positions_diffs[joint_name] = collections.deque()
    # velocities_diffs[joint_name] = collections.deque()
    positions_diffs[joint_name] = []
    velocities_diffs[joint_name] = []
    # efforts_diffs[joint_name] = []
for joint_name in joint_names_feedback:
    positions_feedback_interp[joint_name] = np.interp(ts_joint_states, ts_feedback, feedback["actual"]["positions"][joint_name])
    velocities_feedback_interp[joint_name] = np.interp(ts_joint_states, ts_feedback, feedback["actual"]["velocities"][joint_name])
    # accelerations_goal_interp[joint_name] = np.interp(ts_joint_states, ts_goal, accelerations_goal[joint_name])
    for i in range(len(ts_joint_states)):
        # positions_diffs[joint_name].appendleft(positions_joint_states[joint_name][i] - positions_goal[joint_name][i])
        # velocities_diffs[joint_name].appendleft(velocities_joint_states[joint_name][i] - velocities_goal[joint_name][i])
        positions_diffs[joint_name].append(positions_joint_states[joint_name][i] - positions_feedback_interp[joint_name][i])
        velocities_diffs[joint_name].append(velocities_joint_states[joint_name][i] - velocities_feedback_interp[joint_name][i])
        # efforts_diffs[joint_name].append(efforts_joint_states[joint_name][i] - accelerations_goal_interp[joint_name][i])

print("processing graphs")
n = 0
joint_name = "panda_joint1" 
for joint_name in joint_names_feedback:
# for i in range(1):
    r = 1
    n += 1
    plt.figure(folder_name + " " + str(n))
    for d in ["positions", "velocities"]:
        plt.subplot(2, 2, r)
        plt.plot(ts_feedback, feedback["desired"][d][joint_name], 'o-g', label="desired " + str(d))
        plt.plot(ts_feedback, feedback["actual"][d][joint_name], 'x-b', label="actual " + str(d))
        plt.bar(ts_feedback, feedback["error"][d][joint_name], color="red", edgecolor="red", label="error " + str(d))
        plt.ylabel(str(d))
        plt.title(joint_name)
        plt.legend()
        r += 2
        
    plt.subplot(2, 2, 2)
    # plt.plot(ts_feedback, feedback["actual"]["positions"][joint_name], 'o-g', label="actual positions")
    # plt.plot(ts_feedback, feedback["desired"]["positions"][joint_name], 'o-g', label="desired positions")
    plt.plot(ts_goal, positions_goal[joint_name], 'o-g', label="goal positions")
    plt.plot(ts_joint_states, positions_joint_states[joint_name], 'x-r', label="joint_states positions")
    # plt.bar(ts_joint_states, positions_diffs[joint_name], color="black", edgecolor="black", label="positions diffs")
    plt.ylabel("positions")
    plt.title(joint_name)
    plt.legend()
    plt.subplot(2, 2, 4)
    # plt.plot(ts_feedback, feedback["actual"]["velocities"][joint_name], 'o-g', label="actual velocities")
    # plt.plot(ts_feedback, feedback["desired"]["velocities"][joint_name], 'o-g', label="desired velocities")
    plt.plot(ts_goal, velocities_goal[joint_name], 'o-g', label="goal velocities")
    plt.plot(ts_joint_states, velocities_joint_states[joint_name], 'x-r', label="joint_states velocities")
    plt.plot(ts_joint_states, efforts_joint_states[joint_name], 'x-b', label="joint_states efforts")
    # plt.bar(ts_joint_states, velocities_diffs[joint_name], color="black", edgecolor="black", label="velocities diffs")
    plt.ylabel("velocities")
    plt.title(joint_name)
    plt.legend()

    plt.xlabel('time (ns)')
    plt.legend()

    # plt.figure("efforts " + str(n))
    # plt.subplot(1, 1, 1)
    # plt.title(joint_name)
    # plt.plot(ts_joint_states, efforts_joint_states[joint_name], 'x-b', label="joint_states efforts")
    # plt.ylabel("efforts")
    # plt.xlabel('time (ns)')
    # plt.legend()

# =============================================================================
    # plt.figure(folder_name + " " + str(n))
    # plt.subplot(3, 1, 1)
    # plt.plot(ts_goal, positions_goal[joint_name], 'o-g', label="goal positions")
    # plt.plot(ts_joint_states, positions_joint_states[joint_name], 'x-b', label="joint_states positions")
    # # plt.plot(ts_joint_states, positions_goal_interp[joint_name], 'x-b')
    # plt.bar(ts_joint_states, positions_diffs[joint_name], color="red", edgecolor="red", label="difference")
    # plt.title(joint_name)
    # plt.ylabel('positions')
    # plt.legend()

    # plt.subplot(3, 1, 2)
    # plt.plot(ts_goal, velocities_goal[joint_name], 'o-g', label="goal velocities")
    # plt.plot(ts_joint_states, velocities_joint_states[joint_name], 'x-b', label="joint_states velocities")
    # # plt.plot(ts_joint_states, velocities_goal_interp[joint_name], 'x-b')
    # plt.bar(ts_joint_states, velocities_diffs[joint_name], color="red", edgecolor="red", label="difference")
    # # plt.xlabel('time (s)')
    # plt.ylabel('velocities')
    # plt.legend()

    # plt.subplot(3, 1, 3)
    # plt.plot(ts_goal, accelerations_goal[joint_name], 'o-g', label="goal accelerations")
    # plt.plot(ts_joint_states, efforts_joint_states[joint_name], 'x-b', label="joint_states efforts")
    # plt.bar(ts_joint_states, efforts_diffs[joint_name], color="red", edgecolor="red", label="difference")
    # plt.xlabel('time (s)')
    # plt.ylabel('accelerations/efforts')
    # plt.legend()

print("processing completed")
print("showing graphs")
plt.show()

# for joint_name in joint_names:
#     plt.plot(ts, positions[joint_name], marker='o', linewidth=2)
# plt.show()