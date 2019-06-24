#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

import json, ast, collections, sys, getopt, os, math

scale = 1
input_folder = "latest"
filter_alpha = 0.9
# if len(sys.arvg) == 2:
#     input_folder = sys.argv[1]

def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

try:
    opts, args = getopt.getopt(sys.argv[1:],"i:",["input="])
except getopt.GetoptError:
    print("test.py -i <input_folder>")
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print("test.py -i <input_folder>")
        sys.exit()
    elif opt in ("-i", "--input"):
        input_folder = arg

script_path = os.path.abspath(__file__)
# main_dir = script_path[:script_path.rfind('/utils')]
main_dir = script_path[:script_path.rfind('/scripts')]

# with open("/home/ilbetzy/orebro/trajectory_generation_ws/generated_trajectories/" + input_folder + "/trajectories.txt", 'r') as f:
with open(main_dir + "/generated_trajectories/cpp/" + input_folder + "/trajectories.txt", 'r') as f:
    data = f.read()
trajectories = json.loads(data)
trajectories = ast.literal_eval(json.dumps(trajectories))

joint_trajectories = {}
for i in range(len(trajectories["joint_trajectory"][0])):
    joint_trajectories[i] = []
for values in trajectories["joint_trajectory"]:
    for i in range(len(values)):
        joint_trajectories[i].append(values[i])
    
joint_velocities = {}
for joint_index in range(len(trajectories["joint_trajectory"][0])):
    joint_velocities[joint_index] = []
for i in range(len(trajectories["joint_trajectory"])-1):
    for joint_index in range(len(values)):
        joint_velocities[joint_index].append(trajectories["joint_trajectory"][i+1][joint_index] - trajectories["joint_trajectory"][i][joint_index])
# for i in range(len(trajectories["joint_trajectory"])-1):
#     joint_velocities.append(list(map(lambda j1, j2: j2-j1, trajectories["joint_trajectory"][i], trajectories["joint_trajectory"][i+1])))
for joint_index in range(len(values)):
    joint_velocities[joint_index].append(0)
    
eef_pose = {
    "origin": {
        "x": [],
        "y": [],
        "z": []
    },
    "orientation": {
    }
}
eef_distances = {
    "euclidean_distances": {
        "x": [],
        "y": [],
        "z": [],
        "magnitude": []
    }
}

# eef_distances["euclidean_distances"]["x"][0] = 0
# eef_distances["euclidean_distances"]["y"][0] = 0
# eef_distances["euclidean_distances"]["z"][0] = 0

# eef_distances["euclidean_distances"]["x"].append(0)
# eef_distances["euclidean_distances"]["y"].append(0)
# eef_distances["euclidean_distances"]["z"].append(0)

# don't filter the second distances??
# eef_distances["euclidean_distances"]["x"][1] = (trajectories["eef_trajectory"][1]["origin"]["x"] - trajectories["eef_trajectory"][0]["origin"]["x"])
# eef_distances["euclidean_distances"]["y"][1] = (trajectories["eef_trajectory"][1]["origin"]["y"] - trajectories["eef_trajectory"][0]["origin"]["y"])
# eef_distances["euclidean_distances"]["z"][1] = (trajectories["eef_trajectory"][1]["origin"]["z"] - trajectories["eef_trajectory"][0]["origin"]["z"])

# eef_distances["euclidean_distances"]["x"].append(trajectories["eef_trajectory"][1]["origin"]["x"] - trajectories["eef_trajectory"][0]["origin"]["x"])
# eef_distances["euclidean_distances"]["y"].append(trajectories["eef_trajectory"][1]["origin"]["y"] - trajectories["eef_trajectory"][0]["origin"]["y"])
# eef_distances["euclidean_distances"]["z"].append(trajectories["eef_trajectory"][1]["origin"]["z"] - trajectories["eef_trajectory"][0]["origin"]["z"])

# *100 converts the distances in centimeters
# eef_distances["euclidean_distances"]["x"].append((trajectories["eef_trajectory"][1]["origin"]["x"] - trajectories["eef_trajectory"][0]["origin"]["x"]) * 100)
# eef_distances["euclidean_distances"]["y"].append((trajectories["eef_trajectory"][1]["origin"]["y"] - trajectories["eef_trajectory"][0]["origin"]["y"]) * 100)
# eef_distances["euclidean_distances"]["z"].append((trajectories["eef_trajectory"][1]["origin"]["z"] - trajectories["eef_trajectory"][0]["origin"]["z"]) * 100)
eef_distances["euclidean_distances"]["x"].append(trajectories["eef_trajectory"][1]["origin"]["x"] - trajectories["eef_trajectory"][0]["origin"]["x"])
eef_distances["euclidean_distances"]["y"].append(trajectories["eef_trajectory"][1]["origin"]["y"] - trajectories["eef_trajectory"][0]["origin"]["y"])
eef_distances["euclidean_distances"]["z"].append(trajectories["eef_trajectory"][1]["origin"]["z"] - trajectories["eef_trajectory"][0]["origin"]["z"])
eef_distances["euclidean_distances"]["magnitude"].append(math.sqrt(eef_distances["euclidean_distances"]["x"][0]**2 + eef_distances["euclidean_distances"]["y"][0]**2 + eef_distances["euclidean_distances"]["z"][0]**2))

for index, values in enumerate(trajectories["eef_trajectory"]):
    eef_pose["origin"]["x"].append(values["origin"]["x"])
    eef_pose["origin"]["y"].append(values["origin"]["y"])
    eef_pose["origin"]["z"].append(values["origin"]["z"])

    # calculating filtered distances
    if index != 0:
        # eef_distances["euclidean_distances"]["x"].append(((1 - filter_alpha) * eef_distances["euclidean_distances"]["x"][index - 1]) + (filter_alpha * (trajectories["eef_trajectory"][index]["origin"]["x"] - trajectories["eef_trajectory"][index -1]["origin"]["x"])))
        # eef_distances["euclidean_distances"]["y"].append(((1 - filter_alpha) * eef_distances["euclidean_distances"]["y"][index - 1]) + (filter_alpha * (trajectories["eef_trajectory"][index]["origin"]["y"] - trajectories["eef_trajectory"][index -1]["origin"]["y"])))
        # eef_distances["euclidean_distances"]["z"].append(((1 - filter_alpha) * eef_distances["euclidean_distances"]["z"][index - 1]) + (filter_alpha * (trajectories["eef_trajectory"][index]["origin"]["z"] - trajectories["eef_trajectory"][index -1]["origin"]["z"])))

        # eef_distances["euclidean_distances"]["x"].append(((1 - filter_alpha) * eef_distances["euclidean_distances"]["x"][index - 1]) + (filter_alpha * ((trajectories["eef_trajectory"][index]["origin"]["x"] - trajectories["eef_trajectory"][index -1]["origin"]["x"])) * 100))
        # eef_distances["euclidean_distances"]["y"].append(((1 - filter_alpha) * eef_distances["euclidean_distances"]["y"][index - 1]) + (filter_alpha * ((trajectories["eef_trajectory"][index]["origin"]["y"] - trajectories["eef_trajectory"][index -1]["origin"]["y"])) * 100))
        # eef_distances["euclidean_distances"]["z"].append(((1 - filter_alpha) * eef_distances["euclidean_distances"]["z"][index - 1]) + (filter_alpha * ((trajectories["eef_trajectory"][index]["origin"]["z"] - trajectories["eef_trajectory"][index -1]["origin"]["z"])) * 100))
        eef_distances["euclidean_distances"]["x"].append(((1 - filter_alpha) * eef_distances["euclidean_distances"]["x"][index - 1]) + (filter_alpha * (trajectories["eef_trajectory"][index]["origin"]["x"] - trajectories["eef_trajectory"][index -1]["origin"]["x"])))
        eef_distances["euclidean_distances"]["y"].append(((1 - filter_alpha) * eef_distances["euclidean_distances"]["y"][index - 1]) + (filter_alpha * (trajectories["eef_trajectory"][index]["origin"]["y"] - trajectories["eef_trajectory"][index -1]["origin"]["y"])))
        eef_distances["euclidean_distances"]["z"].append(((1 - filter_alpha) * eef_distances["euclidean_distances"]["z"][index - 1]) + (filter_alpha * (trajectories["eef_trajectory"][index]["origin"]["z"] - trajectories["eef_trajectory"][index -1]["origin"]["z"])))
        eef_distances["euclidean_distances"]["magnitude"].append(math.sqrt(eef_distances["euclidean_distances"]["x"][index]**2 + eef_distances["euclidean_distances"]["y"][index]**2 + eef_distances["euclidean_distances"]["z"][index]**2))

        # eef_distances["euclidean_distances"]["x"][index] =  ((1 - filter_alpha) * eef_distances["euclidean_distances"]["x"][index - 1]) + (filter_alpha * (trajectories["eef_trajectory"][index]["origin"]["x"] - trajectories["eef_trajectory"][index -1]["origin"]["x"]))
        # eef_distances["euclidean_distances"]["y"][index] =  ((1 - filter_alpha) * eef_distances["euclidean_distances"]["y"][index - 1]) + (filter_alpha * (trajectories["eef_trajectory"][index]["origin"]["y"] - trajectories["eef_trajectory"][index -1]["origin"]["y"]))
        # eef_distances["euclidean_distances"]["z"][index] =  ((1 - filter_alpha) * eef_distances["euclidean_distances"]["z"][index - 1]) + (filter_alpha * (trajectories["eef_trajectory"][index]["origin"]["z"] - trajectories["eef_trajectory"][index -1]["origin"]["z"]))
        
print ("eef_distances: ", len(eef_distances["euclidean_distances"]["x"]))
print ("trajectories: ", len(trajectories["eef_trajectory"]))
fk_eef_pose = {
    "origin": {
        "x": [],
        "y": [],
        "z": []
    },
    "orientation": {
    }
}
for values in trajectories["fk_eef_trajectory"]:
    fk_eef_pose["origin"]["x"].append(values["origin"]["x"])
    fk_eef_pose["origin"]["y"].append(values["origin"]["y"])
    fk_eef_pose["origin"]["z"].append(values["origin"]["z"])

# =============================================================================
# feedback processing
# joint_names_feedback = list_message_feedback[0].feedback.joint_names
# feedback = {
#     "desired": {
#         "positions": {},
#         "velocities": {},
#         "accelerations": {}
#     },
#     "actual": {
#         "positions": {},
#         "velocities": {},
#         "accelerations": {}
#     },
#     "error": {
#         "positions": {},
#         "velocities": {},
#         "accelerations": {}
#     }
# }
# for t in feedback:
#     for dimension in feedback[t]:
#         for i in range(len(joint_names_feedback)):
#             feedback[t][dimension][joint_names_feedback[i]] = []
# ts_feedback = []

# for message in list_message_feedback:
#     ts_feedback.append((message.header.stamp.secs * 1000000000) + message.header.stamp.nsecs)
#     for t in feedback:
#         tt = getattr(message.feedback, t)
#         for dimension in feedback[t]:
#             d = getattr(tt, dimension)
#             for i in range(len(d)):
#                 feedback[t][dimension][joint_names_feedback[i]].append(d[i])

# # =============================================================================
# # joint_states processing
# joint_names_joint_states = list_message_joint_states[0].name
# positions_joint_states = {}
# velocities_joint_states = {}
# efforts_joint_states = {}
# for i in range(len(joint_names_joint_states)):
#     positions_joint_states[joint_names_joint_states[i]] = []
#     velocities_joint_states[joint_names_joint_states[i]] = []
#     efforts_joint_states[joint_names_joint_states[i]] = []
# ts_joint_states = []

# for message in list_message_joint_states:
#     ts_joint_states.append((message.header.stamp.secs * 1000000000) + message.header.stamp.nsecs)
#     for i in range(len(joint_names_joint_states)):
#         positions_joint_states[joint_names_joint_states[i]].append(message.position[i])
#         velocities_joint_states[joint_names_joint_states[i]].append(message.velocity[i])
#         # efforts_joint_states[joint_names_joint_states[i]].append(message.effort[i])
#         efforts_joint_states[joint_names_joint_states[i]].append(message.effort[i] / float(scale))

# # =============================================================================
# # computing diffs
# positions_diffs = {}
# velocities_diffs = {}
# # efforts_diffs = {}
# # positions_goal_interp = {}
# # velocities_goal_interp = {}
# positions_feedback_interp = {}
# velocities_feedback_interp = {}
# # accelerations_goal_interp = {}
# for joint_name in joint_names_feedback:
#     # positions_diffs[joint_name] = collections.deque()
#     # velocities_diffs[joint_name] = collections.deque()
#     positions_diffs[joint_name] = []
#     velocities_diffs[joint_name] = []
#     # efforts_diffs[joint_name] = []
# for joint_name in joint_names_feedback:
#     positions_feedback_interp[joint_name] = np.interp(ts_joint_states, ts_feedback, feedback["actual"]["positions"][joint_name])
#     velocities_feedback_interp[joint_name] = np.interp(ts_joint_states, ts_feedback, feedback["actual"]["velocities"][joint_name])
#     # accelerations_goal_interp[joint_name] = np.interp(ts_joint_states, ts_goal, accelerations_goal[joint_name])
#     for i in range(len(ts_joint_states)):
#         # positions_diffs[joint_name].appendleft(positions_joint_states[joint_name][i] - positions_goal[joint_name][i])
#         # velocities_diffs[joint_name].appendleft(velocities_joint_states[joint_name][i] - velocities_goal[joint_name][i])
#         positions_diffs[joint_name].append(positions_joint_states[joint_name][i] - positions_feedback_interp[joint_name][i])
#         velocities_diffs[joint_name].append(velocities_joint_states[joint_name][i] - velocities_feedback_interp[joint_name][i])
#         # efforts_diffs[joint_name].append(efforts_joint_states[joint_name][i] - accelerations_goal_interp[joint_name][i])

print("processing graphs")
for i in range(len(trajectories["joint_trajectory"][0])):
    steps = range(len(trajectories["joint_trajectory"]))
    plt.figure(input_folder + " joint_trajectory " + str(i))
    plt.subplot(2, 1, 1)
    # plt.subplot(1, 1, 1)
    plt.plot(steps, joint_trajectories[i], 'o-g', label="joint_trajectory " + str(i))
    plt.ylabel(str(i))
    # plt.title(joint_name)
    plt.legend()
        
    # plt.subplot(2, 1, 2)
    # plt.plot(ts_feedback, feedback["actual"]["positions"][joint_name], 'o-g', label="actual positions")
    # plt.plot(ts_joint_states, positions_joint_states[joint_name], 'x-r', label="joint_states positions")
    # plt.bar(ts_joint_states, positions_diffs[joint_name], color="red", edgecolor="red", label="positions diffs")
    # plt.ylabel("positions")
    # plt.title(joint_name)
    # plt.legend()

    plt.xlabel('steps')
    plt.legend()


    # plt.figure(input_folder + " joint_velocity " + str(i))
    plt.subplot(2, 1, 2)
    plt.plot(steps, joint_velocities[i], 'o-g', label="joint_velocity " + str(i))
    plt.ylabel(str(i))
    plt.legend()
    plt.xlabel('steps')
    plt.legend()


for axis in eef_distances["euclidean_distances"]:
    steps = range(len(eef_distances["euclidean_distances"][axis]))
    plt.figure(input_folder + " cartesian distances " + str(axis))

    plt.subplot(2, 1, 1)
    # plt.subplot(1, 1, 1)
    plt.plot(steps, eef_distances["euclidean_distances"][axis], 'o-g', label= str(axis) + " axis")
    plt.ylabel(str(axis) + " [m]")
    plt.xlabel("frame count")

    plt.subplot(2, 1, 2)
    num_bins = 10
    n, bins, patches = plt.hist(eef_distances["euclidean_distances"][axis], num_bins)
    plt.xlabel(str(axis) + " [m]")
    plt.ylabel("frame count")

    # plt.title(joint_name)
    plt.legend()


# # ============================3D EEF POSE PLOT=================================
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure("eef_trajectory " + input_folder)

ax = fig.gca(projection='3d')
ax.set_aspect('equal')

# ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(eef_pose["origin"]["x"], eef_pose["origin"]["y"], eef_pose["origin"]["z"], c=eef_pose["origin"]["z"], cmap='Greens', label="eef_trajectory")

# fig = plt.figure("fk_eef_trajectory")
# ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(fk_eef_pose["origin"]["x"], fk_eef_pose["origin"]["y"], fk_eef_pose["origin"]["z"], c=fk_eef_pose["origin"]["z"], marker='^', cmap='Reds', label="fk_eef_trajectory")
plt.legend()
# ax.axis('equal')

# # =============================================================================
#     # plt.figure(input_folder + " " + str(n))
#     # plt.subplot(3, 1, 1)
#     # plt.plot(ts_goal, positions_goal[joint_name], 'o-g', label="goal positions")
#     # plt.plot(ts_joint_states, positions_joint_states[joint_name], 'x-b', label="joint_states positions")
#     # # plt.plot(ts_joint_states, positions_goal_interp[joint_name], 'x-b')
#     # plt.bar(ts_joint_states, positions_diffs[joint_name], color="red", edgecolor="red", label="difference")
#     # plt.title(joint_name)
#     # plt.ylabel('positions')
#     # plt.legend()

#     # plt.subplot(3, 1, 2)
#     # plt.plot(ts_goal, velocities_goal[joint_name], 'o-g', label="goal velocities")
#     # plt.plot(ts_joint_states, velocities_joint_states[joint_name], 'x-b', label="joint_states velocities")
#     # # plt.plot(ts_joint_states, velocities_goal_interp[joint_name], 'x-b')
#     # plt.bar(ts_joint_states, velocities_diffs[joint_name], color="red", edgecolor="red", label="difference")
#     # # plt.xlabel('time (s)')
#     # plt.ylabel('velocities')
#     # plt.legend()

#     # plt.subplot(3, 1, 3)
#     # plt.plot(ts_goal, accelerations_goal[joint_name], 'o-g', label="goal accelerations")
#     # plt.plot(ts_joint_states, efforts_joint_states[joint_name], 'x-b', label="joint_states efforts")
#     # plt.bar(ts_joint_states, efforts_diffs[joint_name], color="red", edgecolor="red", label="difference")
#     # plt.xlabel('time (s)')
#     # plt.ylabel('accelerations/efforts')
#     # plt.legend()

set_axes_equal(ax)
print("processing completed")
print("showing graphs")
plt.show()

# for joint_name in joint_names:
#     plt.plot(ts, positions[joint_name], marker='o', linewidth=2)
# plt.show()