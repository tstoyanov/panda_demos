import argparse
import json, ast, os, math
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser(description='Joints position dataset generator')
parser.add_argument('-i', default="/home/ilbetzy/orebro/src/panda_demos/trajectory_generator/generated_trajectories/cpp/fourth_batch",
                    help='path of the input folder')
args = parser.parse_args()
input_folder = args.i

trajectory_files = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]
joint_trajectories_dataset = []
joint_trajectories_dataset = {
    "joints_positions": [],
    "eef_velocity_magnitude": []
}

for n, trajectory_file in enumerate(trajectory_files):
    with open(input_folder + "/" + trajectory_file, 'r') as f:
        data = f.read()
    trajectories = json.loads(data)
    joint_trajectories_dataset["joints_positions"].append(trajectories["joint_trajectory"])
    
    trajectories = ast.literal_eval(json.dumps(trajectories))
    filter_alpha = 0.9
    filter_depth = 3
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

    joint_trajectories_dataset["eef_velocity_magnitude"].append(euclidean_distances["magnitude"])
    #     release_distances["euclidean_distances"][key].append(euclidean_distances[key])
    
    if n % 100 == 99:
        print ("files scanned: ", n+1)


script_path = os.path.abspath(__file__)
main_dir = script_path[:script_path.rfind('/utils')]
dataset_dir = main_dir + "/generated_trajectories/datasets"
dataset_file = dataset_dir + input_folder[input_folder.rindex("/"):] + "/dataset.txt"

os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
with open(dataset_file, "w") as f:
    json.dump(joint_trajectories_dataset, f)