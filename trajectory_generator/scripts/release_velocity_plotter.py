#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

import json, ast, collections, sys, getopt, os, math

scale = 1
input_folder = "latest"
batch = False
filter_alpha = 0.9
execution_time = False

try:
    opts, args = getopt.getopt(sys.argv[1:],"i:b:t:",["input=", "batch=", "time="])
except getopt.GetoptError:
    print("test.py -i <input_folder>")
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print("test.py -i <input_folder>")
        sys.exit()
    elif opt in ("-i", "--input"):
        input_folder = arg
    elif opt in ("-b", "--batch"):
        input_folder = arg
        batch = True
    elif opt in ("-t", "--time"):
        execution_time = float(arg) / 1000000000 # converting from nano seconds to seconds


script_path = os.path.abspath(__file__)
# main_dir = script_path[:script_path.rfind('/utils')]
main_dir = script_path[:script_path.rfind('/scripts')]

if batch:
    # =============================================================================
    batch_folder = main_dir + "/generated_trajectories/cpp/" + input_folder
    batch_trajectories = [f for f in listdir(batch_folder) if isfile(join(batch_folder, f))]
    filter_depth = 3
    release_distances = {
        "euclidean_distances": {
            "x": [],
            "y": [],
            "z": [],
            "magnitude": [],
        }
    }
    files_computed = 0
    print("processing data...")
    for batch_trajectory_file in batch_trajectories:
        with open(batch_folder + "/" + batch_trajectory_file, 'r') as f:
            data = f.read()
        trajectories = json.loads(data)
        trajectories = ast.literal_eval(json.dumps(trajectories))

        if execution_time != False:
            dt = execution_time / len(trajectories["joint_trajectory"])

        try:
            realease_frame = trajectories["realease_frame"] - 1
        except:
            realease_frame = 90 - 1
        
        euclidean_distances = {
            # *100 converts the distances in centimeters
            "x": (trajectories["eef_trajectory"][realease_frame - filter_depth]["origin"]["x"] - trajectories["eef_trajectory"][realease_frame-filter_depth-1]["origin"]["x"]) * 100,
            "y": (trajectories["eef_trajectory"][realease_frame - filter_depth]["origin"]["y"] - trajectories["eef_trajectory"][realease_frame-filter_depth-1]["origin"]["y"]) * 100,
            "z": (trajectories["eef_trajectory"][realease_frame - filter_depth]["origin"]["z"] - trajectories["eef_trajectory"][realease_frame-filter_depth-1]["origin"]["z"]) * 100,
            "magnitude": 0
        }
        for i in reversed(range(filter_depth)):
            euclidean_distances["x"] = ((1 - filter_alpha) * euclidean_distances["x"]) + (filter_alpha * ((trajectories["eef_trajectory"][realease_frame-i]["origin"]["x"] - trajectories["eef_trajectory"][realease_frame-i-1]["origin"]["x"]) * 100))
            euclidean_distances["y"] = ((1 - filter_alpha) * euclidean_distances["y"]) + (filter_alpha * ((trajectories["eef_trajectory"][realease_frame-i]["origin"]["y"] - trajectories["eef_trajectory"][realease_frame-i-1]["origin"]["y"]) * 100))
            euclidean_distances["z"] = ((1 - filter_alpha) * euclidean_distances["z"]) + (filter_alpha * ((trajectories["eef_trajectory"][realease_frame-i]["origin"]["z"] - trajectories["eef_trajectory"][realease_frame-i-1]["origin"]["z"]) * 100))
        euclidean_distances["magnitude"] = math.sqrt(euclidean_distances["x"]**2 + euclidean_distances["y"]**2 + euclidean_distances["z"]**2)

        for key in euclidean_distances:
            release_distances["euclidean_distances"][key].append(euclidean_distances[key])
        
        files_computed += 1
        print ("files_computed = ", files_computed)
            
    print("data processing completed")
    print("processing graphs")

    plt.figure(input_folder + " euclidean distances")
    
    final_xlim = [0, 0]
    for index, axis in enumerate(release_distances["euclidean_distances"]):
        plt.subplot(2, 2, index+1)
        num_bins = 100
        if execution_time != False:
            n, bins, patches = plt.hist([float(item) / dt for item in release_distances["euclidean_distances"][axis]], num_bins, edgecolor='blue')
            plt.xlabel(str(axis) + " [cm/s]")
        else:
            n, bins, patches = plt.hist(release_distances["euclidean_distances"][axis], num_bins, edgecolor='blue')
            plt.xlabel(str(axis) + " [cm/dt]")
        current_xlim = plt.xlim()
        final_xlim[0] = min(final_xlim[0], current_xlim[0])
        final_xlim[1] = max(final_xlim[1], current_xlim[1])
        plt.ylabel("frame count")
        plt.legend()
    
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.xlim(final_xlim[0], final_xlim[1])
    
    print("graphs processing completed")
    print("showing graphs")
    plt.show()
# =============================================================================

else:
    # with open("/home/ilbetzy/orebro/trajectory_generation_ws/generated_trajectories/" + input_folder + "/trajectories.txt", 'r') as f:
    with open(main_dir + "/generated_trajectories/cpp/" + input_folder + "/trajectories.txt", 'r') as f:
        data = f.read()
    trajectories = json.loads(data)
    trajectories = ast.literal_eval(json.dumps(trajectories))

    if execution_time != False:
        dt = execution_time / len(trajectories["joint_trajectory"])

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
    
    joint_accelerations = {}
    for joint_index in range(len(trajectories["joint_trajectory"][0])):
        joint_accelerations[joint_index] = []
    for i in range(len(trajectories["joint_trajectory"])-1):
        for joint_index in range(len(values)):
            joint_accelerations[joint_index].append(trajectories["joint_trajectory"][i+1][joint_index] - trajectories["joint_trajectory"][i][joint_index])
    # for i in range(len(trajectories["joint_trajectory"])-1):
    #     joint_accelerations.append(list(map(lambda j1, j2: j2-j1, trajectories["joint_trajectory"][i], trajectories["joint_trajectory"][i+1])))
    for joint_index in range(len(values)):
        joint_accelerations[joint_index].append(0)

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

    # *100 converts the distances in centimeters
    eef_distances["euclidean_distances"]["x"].append((trajectories["eef_trajectory"][1]["origin"]["x"] - trajectories["eef_trajectory"][0]["origin"]["x"]) * 100)
    eef_distances["euclidean_distances"]["y"].append((trajectories["eef_trajectory"][1]["origin"]["y"] - trajectories["eef_trajectory"][0]["origin"]["y"]) * 100)
    eef_distances["euclidean_distances"]["z"].append((trajectories["eef_trajectory"][1]["origin"]["z"] - trajectories["eef_trajectory"][0]["origin"]["z"]) * 100)
    eef_distances["euclidean_distances"]["magnitude"].append(math.sqrt(eef_distances["euclidean_distances"]["x"][0]**2 + eef_distances["euclidean_distances"]["y"][0]**2 + eef_distances["euclidean_distances"]["z"][0]**2))

    for index, values in enumerate(trajectories["eef_trajectory"]):
        eef_pose["origin"]["x"].append(values["origin"]["x"])
        eef_pose["origin"]["y"].append(values["origin"]["y"])
        eef_pose["origin"]["z"].append(values["origin"]["z"])

        # calculating filtered distances
        if index != 0:     
            eef_distances["euclidean_distances"]["x"].append(((1 - filter_alpha) * eef_distances["euclidean_distances"]["x"][index - 1]) + (filter_alpha * ((trajectories["eef_trajectory"][index]["origin"]["x"] - trajectories["eef_trajectory"][index -1]["origin"]["x"])) * 100))
            eef_distances["euclidean_distances"]["y"].append(((1 - filter_alpha) * eef_distances["euclidean_distances"]["y"][index - 1]) + (filter_alpha * ((trajectories["eef_trajectory"][index]["origin"]["y"] - trajectories["eef_trajectory"][index -1]["origin"]["y"])) * 100))
            eef_distances["euclidean_distances"]["z"].append(((1 - filter_alpha) * eef_distances["euclidean_distances"]["z"][index - 1]) + (filter_alpha * ((trajectories["eef_trajectory"][index]["origin"]["z"] - trajectories["eef_trajectory"][index -1]["origin"]["z"])) * 100))
            eef_distances["euclidean_distances"]["magnitude"].append(math.sqrt(eef_distances["euclidean_distances"]["x"][index]**2 + eef_distances["euclidean_distances"]["y"][index]**2 + eef_distances["euclidean_distances"]["z"][index]**2))
        
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

    print("processing graphs")
    for i in range(len(trajectories["joint_trajectory"][0])):
        steps = range(len(trajectories["joint_trajectory"]))
        plt.figure(input_folder + " - joint_" + str(i+1))
        if execution_time != False:
            plt.subplot(2, 1, 2)
            plt.plot(steps, [float(item) / dt for item in joint_velocities[i]], 'o-g', label="joint_" + str(i+1) + " velocity")
            plt.ylabel("v [rad/s]")
            plt.legend()
            plt.xlabel('steps')
            plt.legend()
            plt.subplot(2, 1, 1)
        else:
            plt.subplot(1, 1, 1)
        plt.plot(steps, joint_trajectories[i], 'o-g', label="joint_" + str(i+1) + " trajectory")
        plt.ylabel("pos [rad]")
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


    for axis in eef_distances["euclidean_distances"]:
        steps = range(len(eef_distances["euclidean_distances"][axis]))
        plt.figure(input_folder + " euclidean distances " + str(axis))

        plt.subplot(2, 1, 1)
        if execution_time != False:
            plt.plot(steps, [float(item) / dt for item in eef_distances["euclidean_distances"][axis]], 'o-g', label= str(axis) + " axis")
            plt.ylabel(str(axis) + " [cm/s]")
        else:
            plt.plot(steps, eef_distances["euclidean_distances"][axis], 'o-g', label= str(axis) + " axis")
            plt.ylabel(str(axis) + " [cm/dt]")
        plt.xlabel("frame count")

        plt.subplot(2, 1, 2)
        num_bins = 10
        if execution_time != False:
            n, bins, patches = plt.hist([float(item) / dt for item in eef_distances["euclidean_distances"][axis]], num_bins)
            plt.xlabel(str(axis) + " [cm/s]")
        else:
            n, bins, patches = plt.hist(eef_distances["euclidean_distances"][axis], num_bins)
            plt.xlabel(str(axis) + " [cm/dt]")
        plt.ylabel("frame count")

        # plt.title(joint_name)
        plt.legend()


    # # ============================3D EEF POSE PLOT=================================
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure("eef_trajectory " + input_folder)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(eef_pose["origin"]["x"], eef_pose["origin"]["y"], eef_pose["origin"]["z"], c=eef_pose["origin"]["z"], cmap='Greens', label="eef_trajectory")

    # fig = plt.figure("fk_eef_trajectory")
    # ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(fk_eef_pose["origin"]["x"], fk_eef_pose["origin"]["y"], fk_eef_pose["origin"]["z"], c=fk_eef_pose["origin"]["z"], marker='^', cmap='Reds', label="fk_eef_trajectory")
    plt.legend()
    ax.axis('equal')

    print("processing completed")
    print("showing graphs")
    plt.show()