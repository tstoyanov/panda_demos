#!/usr/bin/env python
import json
import copy
import rospy
from std_msgs.msg import Header
from std_msgs.msg import String
from actionlib_msgs.msg import GoalID
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryGoal
from control_msgs.msg import FollowJointTrajectoryActionGoal
from moveit_msgs.msg import ExecuteTrajectoryGoal
from moveit_msgs.msg import ExecuteTrajectoryActionGoal

from rospy_message_converter import json_message_converter

import ast
import sys
import moveit_commander
import getopt

input_folder = "test"
# tot_time_nsecs = 10000000000  # total execution time for the trajectory in nanoseconds
tot_time_nsecs = 3000000000  # total execution time for the trajectory in nanoseconds

try:
    opts, args = getopt.getopt(sys.argv[1:],"i:t:",["input=", "nanoseconds="])
except getopt.GetoptError:
    print("test.py -i <input_folder> -t <trajectory_execution_time>")
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print("test.py -i <input_folder> -t <trajectory_execution_time>")
        sys.exit()
    elif opt in ("-i", "--input"):
        folder_name = arg
    elif opt in ("-t", "--nanoseconds"):
        tot_time_nsecs = int(arg)

print("input_folder = " + str(input_folder))

def talker():
    pub = rospy.Publisher('/position_joint_trajectory_controller/follow_joint_trajectory/goal',
                          FollowJointTrajectoryActionGoal, queue_size=10)
    rospy.init_node('myWriter', anonymous=True)
    rate = rospy.Rate(0.1)  # hz

# ==========================================================================

    moveit_commander.roscpp_initialize(sys.argv)
    group_name = "panda_arm"
    group = moveit_commander.MoveGroupCommander(group_name)

    joint_goal = group.get_current_joint_values()
    start = [-0.448036147657, 0.328661662868, -0.622003205874, -1.82402771276, 0.269721323163, 2.1145116905, -1.94276850845]
    kdl_start = [-1.15395, 0.612792, 0.155895, -1.93626, 2.8973, 0.607409, -2.8973]
    kdl_fl_start = [0.841403, 0.4, -0.7, -2.22141, -0.25, 1.8, -2.2]
    kdl_average_start = [-0.455889, 0.710596, -0.561466, -1.90657, -2.56471, 0.690288, -2.56108]
    kdl_start_joint_pos_array = [-0.443379, 0.702188, -0.556869, -1.9368, -2.55769, 0.667764, -2.56121]
    for i in range(len(start)):
        # joint_goal[i] = kdl_start_joint_pos_array[i]
        joint_goal[i] = start[i]
    group.go(joint_goal, wait=True)
    group.stop()

    print("=== Press `Enter` to write ===")
    raw_input()

# ==========================================================================
    # getting the generated trajectory data
    # cpp trajectories
    # with open("/home/ilbetzy/orebro/trajectory_generation_ws/generated_trajectories/cpp/" + input_folder + "/trajectories.txt", 'r') as f:
    #     # data = json.load(f)
    #     data = f.read()
    # trajectories = json.loads(data)
    # trajectories = ast.literal_eval(json.dumps(trajectories))

    # joint_trajectories = {}
    # for i in range(len(trajectories["joint_trajectory"][0])):
    #     joint_trajectories[i] = []
    # for values in trajectories["joint_trajectory"]:
    #     for i in range(len(values)):
    #         joint_trajectories[i].append(values[i])


    # eef_pose = {
    #     "x": [],
    #     "y": [],
    #     "z": []
    # }
    # for values in trajectories["eef_trajectory"]:
    #     eef_pose["x"].append(values["x"])
    #     eef_pose["y"].append(values["y"])
    #     eef_pose["z"].append(values["z"])
    
    # base_trajectory_message = json_message_converter.convert_json_to_ros_message('moveit_msgs/ExecuteTrajectoryActionGoal', data)

    # getting the message structure form a real message
    with open("/home/ilbetzy/orebro/joints_check/logs/c_twice_speed_2/json/position_joint_trajectory_controller-follow_joint_trajectory-goal.mylog", 'r') as f:
    # with open("/home/ilbetzy/orebro/joints_check/logs/test_normal_speed/json/position_joint_trajectory_controller-follow_joint_trajectory-goal.mylog", 'r') as f:
        # data = json.load(f)
        data = f.read()
    message = json_message_converter.convert_json_to_ros_message(
        'control_msgs/FollowJointTrajectoryActionGoal', data)

    now = rospy.get_rostime()
    while (now.secs == 0 | now.nsecs == 0):
        now = rospy.get_rostime()

    seq = 0

    # while not rospy.is_shutdown():
    seq += 1
    secs = now.secs
    nsecs = now.nsecs

    message.header.stamp.secs = secs
    message.header.stamp.nsecs = nsecs
    message.goal_id.stamp.secs = secs
    message.goal_id.stamp.nsecs = nsecs
    message.goal_id.id = "/myWriter-"+str(seq)+"-"+str(secs)+"."+str(nsecs)

    # if speed == "half_speed":
    #     # # double the time_from_start attribute -> half the speed
    #     for point in base_trajectory_message.goal.trajectory.joint_trajectory.points:
    #         point.time_from_start.secs *= 2
    #         point.time_from_start.nsecs *= 2
    #         if (point.time_from_start.nsecs > 999999999):
    #             point.time_from_start.secs += 1
    #             point.time_from_start.nsecs -= 1000000000
    # elif speed == "twice_speed":
    #     # halve the time_from_start attribute -> double the speed
    #     for point in base_trajectory_message.goal.trajectory.joint_trajectory.points:
    #         a = point.time_from_start.secs // 2
    #         b = point.time_from_start.secs / 2.0 - a
    #         point.time_from_start.secs = a
    #         point.time_from_start.nsecs = (point.time_from_start.nsecs // 2) + (b * 1000000000)
    #         for i in range(len(point.velocities)):
    #             point.velocities[i] *= 2.0
    #         for i in range(len(point.accelerations)):
    #             point.accelerations[i] *= 4.0
    
    # overwriting the "positions" list in the message structure with the data from the generated trajectory

    # ========== MOVEIT ==========
    with open("/home/ilbetzy/orebro/trajectory_generation_ws/generated_trajectories/moveit/" + input_folder + "/trajectories.txt", 'r') as f:
        # data = json.load(f)
        data = f.read()
    moveit_trajectories = json.loads(data)
    moveit_trajectories = ast.literal_eval(json.dumps(moveit_trajectories))

    trajectory_point = JointTrajectoryPoint()
    print ("trajectory_point: ", trajectory_point)

    temp_points = []
    zero_list = [0]*len(message.goal.trajectory.points[0].positions)
    dt = int(tot_time_nsecs / len(moveit_trajectories["joint_trajectory"]["points"]))
    print ("DT", dt)
    current_time_from_start = {
        "secs": 0,
        "nsecs": 0
    }
    for point in moveit_trajectories["joint_trajectory"]["points"]:
        trajectory_point.positions = point["positions"]
        trajectory_point.time_from_start.secs = current_time_from_start["secs"]
        trajectory_point.time_from_start.nsecs = current_time_from_start["nsecs"]
        print ("trajectory_point: ", trajectory_point)
        temp_points.append(copy.deepcopy(trajectory_point))
        print ("temp_points[-1]: ", temp_points[-1])
        print ("BEFORE current_time_from_start", current_time_from_start)
        current_time_from_start["nsecs"] += dt
        print ("AFTER current_time_from_start", current_time_from_start)
        if current_time_from_start["nsecs"] >= 1000000000:
            # current_time_from_start["nsecs"] -= 1000000000
            # current_time_from_start["secs"] += 1

            current_time_from_start["secs"] += int(current_time_from_start["nsecs"] / 1000000000)
            current_time_from_start["nsecs"] = current_time_from_start["nsecs"] % 1000000000
    # ========== END MOVEIT ==========
    
    # print ('temp_points: ', temp_points)
    raw_input()
    # ========== CPP ==========
    # temp_points = []
    # zero_list = [0]*len(message.goal.trajectory.points[0].positions)
    # for i in range(len(trajectories["joint_trajectory"])):
    #     temp_points.append(my_point(trajectories["joint_trajectory"][i], zero_list, zero_list, [], message.goal.trajectory.points[i].time_from_start))
    # ========== END CPP ==========

    message.goal.trajectory.points = temp_points
    rospy.loginfo(message)
    pub.publish(message)

    # rate.sleep()

class my_point:
    def __init__(self, positions, velocities, accelerations, effort, time_from_start):
        self.positions = positions
        self.velocities = velocities
        self.accelerations = accelerations
        self.effort = effort
        self.time_from_start = time_from_start

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
