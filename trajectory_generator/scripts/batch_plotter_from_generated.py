#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

import json, ast, collections, sys, getopt, os

from os import listdir
from os.path import isfile, join

input_folder = "latest_batch"

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
main_dir = script_path[:script_path.rfind('/utils')]
batch_dir = main_dir + "/generated_trajectories/cpp/" + input_folder
trajectory_files = [f for f in listdir(batch_dir) if isfile(join(batch_dir, f))]

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure("eef_trajectory " + input_folder)
# ax = fig.add_subplot(111, projection='3d')
# ax.axis('equal')
ax = fig.gca(projection='3d')
ax.set_aspect('equal')

n = 0
for trajectory_file in trajectory_files:
    n += 1
    print ("n = ", n)
    with open(main_dir + "/generated_trajectories/cpp/" + input_folder + "/" + trajectory_file, 'r') as f:
        data = f.read()
    trajectories = json.loads(data)
    trajectories = ast.literal_eval(json.dumps(trajectories))

    eef_pose = {
        "origin": {
            "x": [],
            "y": [],
            "z": []
        },
        "orientation": {
        }
    }
    for values in trajectories["eef_trajectory"]:
        eef_pose["origin"]["x"].append(values["origin"]["x"])
        eef_pose["origin"]["y"].append(values["origin"]["y"])
        eef_pose["origin"]["z"].append(values["origin"]["z"])

    release_frame = trajectories["realease_frame"] - 1
    ax.plot(eef_pose["origin"]["x"], eef_pose["origin"]["y"], eef_pose["origin"]["z"], alpha=0.1, color="blue", linewidth=1)
    ax.plot([trajectories["eef_trajectory"][release_frame]["origin"]["x"]], [trajectories["eef_trajectory"][release_frame]["origin"]["y"]], [trajectories["eef_trajectory"][release_frame]["origin"]["z"]], markerfacecolor='red', markeredgecolor='red', marker='o', markersize=2)

set_axes_equal(ax)
# plt.legend()
plt.show()