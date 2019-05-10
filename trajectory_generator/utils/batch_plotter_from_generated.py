#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

import json, ast, collections, sys, getopt, os

from os import listdir
from os.path import isfile, join

input_folder = "latest_batch"

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
ax = fig.add_subplot(111, projection='3d')

n = 0
for trajectory_file in trajectory_files:
    print ("n = ", n)
    n += 1
    print ("trajectory_file: ", trajectory_file)
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

    ax.plot(eef_pose["origin"]["x"], eef_pose["origin"]["y"], eef_pose["origin"]["z"], alpha=0.1, color="blue", linewidth=1)
    ax.axis('equal')

# plt.legend()
plt.show()