#!/usr/bin/env python
import json
import copy
import rospy
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryActionGoal
from franka_gripper.msg import GraspActionGoal
from franka_gripper.msg import MoveActionGoal

import ast
import sys
import moveit_commander
import getopt
import os

import rospkg

rospack = rospkg.RosPack()

input_folder = "latest"
is_simulation = False
tot_time_nsecs = 2000000000  # total execution time for the trajectory in nanoseconds

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

deceleration_frames = 5     # number of frames used to decelerate
deceleration_time = 0.5    # deceleration time in seconds
# deceleration_time = 0.25    # deceleration time in seconds
deceleration_dt = float(deceleration_time) / deceleration_frames

gripper_open_delay = 0.3    # delay in seconds between sending and executing the open command

def talker():

    if is_simulation:
        print("SIMULATION MODE")
        pub = rospy.Publisher('/position_joint_trajectory_controller/follow_joint_trajectory/goal',
                            FollowJointTrajectoryActionGoal, queue_size=10)
        grasp_pub = rospy.Publisher('/franka_gripper/grasp/goal',
                            GraspActionGoal, queue_size=10)
        gripper_move_pub = rospy.Publisher('/franka_gripper/move/goal',
                            MoveActionGoal, queue_size=10)
        # getting the generated trajectory data
        package_path = rospack.get_path("trajectory_generator")
        with open(package_path + "/generated_trajectories/cpp/" + input_folder + "/trajectories.txt", 'r') as f:
            data = f.read()
    else:
        pub = rospy.Publisher('/panda/position_joint_trajectory_controller/follow_joint_trajectory/goal',
                            FollowJointTrajectoryActionGoal, queue_size=10)
        grasp_pub = rospy.Publisher('/panda/franka_gripper/grasp/goal',
                            GraspActionGoal, queue_size=10)
        gripper_move_pub = rospy.Publisher('/panda/franka_gripper/move/goal',
                            MoveActionGoal, queue_size=10)
        with open(input_folder, 'r') as f:
            data = f.read()

    rospy.init_node('myWriter', anonymous=True)
    rate = rospy.Rate(0.1)  # hz

    moveit_commander.roscpp_initialize(sys.argv)
    group_name = "panda_arm"
    group = moveit_commander.MoveGroupCommander(group_name)

    joint_goal = group.get_current_joint_values()
    start = [-0.448036147657, 0.328661662868, -0.622003205874, -1.82402771276, 0.269721323163, 2.1145116905, -1.94276850845]
    kdl_start = [-1.15395, 0.612792, 0.155895, -1.93626, 2.8973, 0.607409, -2.8973]
    kdl_fl_start = [0.841403, 0.4, -0.7, -2.22141, -0.25, 1.8, -2.2]
    kdl_average_start = [-0.455889, 0.710596, -0.561466, -1.90657, -2.56471, 0.690288, -2.56108]
    kdl_start_joint_pos_array = [-0.443379, 0.702188, -0.556869, -1.9368, -2.55769, 0.667764, -2.56121]
    # for i in range(len(start)):
    #     joint_goal[i] = start[i]
    # group.go(joint_goal, wait=True)
    # group.stop()

    
    trajectories = json.loads(data)
    trajectories = ast.literal_eval(json.dumps(trajectories))
    joint_trajectories = {}
    for i in range(len(trajectories["joint_trajectory"][0])):
        joint_trajectories[i] = []
    for values in trajectories["joint_trajectory"]:
        for i in range(len(values)):
            joint_trajectories[i].append(values[i])
    joint_names = trajectories["joint_names"]
    # eef_pose = {
    #     "origin":
    #     {
    #         "x": [],
    #         "y": [],
    #         "z": [],
    #     },
    #     "orientation":
    #     {
    #     }
    # }
    # for values in trajectories["eef_trajectory"]:
    #     print ("values: ", values)
    #     eef_pose["origin"]["x"].append(values["origin"]["x"])
    #     eef_pose["origin"]["y"].append(values["origin"]["y"])
    #     eef_pose["origin"]["z"].append(values["origin"]["z"])

    # go to the real initial point and open the gripper
    gripper_move_message = MoveActionGoal()
    gripper_move_message.goal.width = 0.09
    gripper_move_message.goal.speed = 0.05
    gripper_move_pub.publish(gripper_move_message)
    
    for i in range(len(joint_trajectories)):
        joint_goal[i] = joint_trajectories[i][0]
    group.go(joint_goal, wait=True)
    group.stop()

    dt = int(tot_time_nsecs / len(trajectories["joint_trajectory"]))
    trajectories["joint_velocity"] = []
    trajectories["joint_acceleration"] = []
    trajectories["ds"] = []
    trajectories["dv"] = []
    # for i in range(len(trajectories["joint_trajectory"])-1):
    #     trajectories["ds"].append(list(map(lambda s1, s2: s2 - s1, trajectories["joint_trajectory"][i], trajectories["joint_trajectory"][i+1])))
    #     trajectories["joint_velocity"].append(list(map(lambda ds: ds*1000000000/dt * 1, trajectories["ds"][i])))
    #     # trajectories["joint_velocity"].append(list(map(lambda s1, s2: (s2-s1)*1000000000/dt * 1.1, trajectories["joint_trajectory"][i], trajectories["joint_trajectory"][i+1])))    
    # trajectories["joint_velocity"].append([0]*len(trajectories["joint_trajectory"][0])) 
    for i in range(len(trajectories["joint_trajectory"])-1):
        trajectories["ds"].append(list(map(lambda s1, s2: s2 - s1, trajectories["joint_trajectory"][i], trajectories["joint_trajectory"][i+1])))
        trajectories["joint_velocity"].append(list(map(lambda ds: ds*1000000000/dt * 1, trajectories["ds"][i])))
        # trajectories["joint_velocity"].append(list(map(lambda s1, s2: (s2-s1)*1000000000/dt * 1.1, trajectories["joint_trajectory"][i], trajectories["joint_trajectory"][i+1])))    
    # trajectories["joint_velocity"].append([0]*len(trajectories["joint_trajectory"][0]))
    trajectories["ds"].append(list(map(lambda s1, s2: s2 - s1, trajectories["joint_trajectory"][-2], trajectories["joint_trajectory"][-1])))
    trajectories["joint_velocity"].append(list(map(lambda ds: ds*1000000000/dt * 1, trajectories["ds"][-1])))

    
    # for i in range(len(trajectories["joint_trajectory"])-1):        
    #     trajectories["dv"].append(list(map(lambda v1, v2: v2 - v1, trajectories["joint_velocity"][i], trajectories["joint_velocity"][i+1])))
    #     trajectories["joint_acceleration"].append(list(map(lambda dv: dv*1000000000/dt * 1, trajectories["dv"][i])))
    #     # trajectories["joint_velocity"].append(list(map(lambda s1, s2: (s2-s1)*1000000000/dt * 1.1, trajectories["joint_trajectory"][i], trajectories["joint_trajectory"][i+1])))
    # trajectories["joint_acceleration"].append([0]*len(trajectories["joint_trajectory"][0]))
    for i in range(len(trajectories["joint_trajectory"])-1):        
        trajectories["dv"].append(list(map(lambda v1, v2: v2 - v1, trajectories["joint_velocity"][i], trajectories["joint_velocity"][i+1])))
        trajectories["joint_acceleration"].append(list(map(lambda dv: dv*1000000000/dt * 1, trajectories["dv"][i])))
        # trajectories["joint_velocity"].append(list(map(lambda s1, s2: (s2-s1)*1000000000/dt * 1.1, trajectories["joint_trajectory"][i], trajectories["joint_trajectory"][i+1])))
    trajectories["dv"].append(list(map(lambda v1, v2: v2 - v1, trajectories["joint_velocity"][-2], trajectories["joint_velocity"][-1])))
    trajectories["joint_acceleration"].append(list(map(lambda dv: dv*1000000000/dt * 1, trajectories["dv"][-1])))

    def deceleration_positions (v1, v2, initial_pos):
        return initial_pos + (v1+v2)/2 * deceleration_dt

    deceleration = {
        "joints_positions": [],
        "joints_velocities": []
    }
    for i in range(deceleration_frames):
        deceleration["joints_velocities"].append(list(map(lambda release_velocity: release_velocity*(deceleration_frames - (i+1))/deceleration_frames, trajectories["joint_velocity"][-1])))
    for i in range(deceleration_frames):
        if i == 0:
            deceleration["joints_positions"].append(list(map(lambda vel1, vel2, initial_position: initial_position + (vel1+vel2)/2 * deceleration_dt, deceleration["joints_velocities"][i], trajectories["joint_velocity"][-1], trajectories["joint_trajectory"][-1])))
        else:
            deceleration["joints_positions"].append(list(map(lambda vel1, vel2, initial_position: initial_position + (vel1+vel2)/2 * deceleration_dt, deceleration["joints_velocities"][i], deceleration["joints_velocities"][i-1], deceleration["joints_positions"][-1])))
        # deceleration["joints_positions"] = trajectories["ds"][-1]
        # deceleration["joints_velocities"] = trajectories["joint_trajectory"]
    print (deceleration["joints_velocities"])
    print (deceleration["joints_positions"])
    
    for i in range(len(trajectories["joint_velocity"])):
        print ('trajectories["joint_velocity"][' + str(i) + '] = ' + str(trajectories["joint_velocity"][i]))

    # print("=== Press `Enter` to grasp ===")
    print("=== Press `Enter` to position the gripper ===")
    raw_input()
    grasp_message = GraspActionGoal()
    grasp_message.goal.width = 0.02
    grasp_message.goal.epsilon.inner = 0.01
    grasp_message.goal.epsilon.outer = 0.01
    grasp_message.goal.speed = 0.05
    grasp_message.goal.force = 0.01

    grasp_pub.publish(grasp_message)

    # ===== Sliding throw =====
    # gripper_move_message = MoveActionGoal()
    # gripper_move_message.goal.width = 0.01
    # gripper_move_message.goal.speed = 0.05
    # gripper_move_pub.publish(gripper_move_message)

    print("=== Press `Enter` to print trajectory ===")
    raw_input()

    release_time_from_start = {
        "secs": 0,
        "nsecs": 0
    }
    adjusted_release_time = {
        "secs": 0,
        "nsecs": 0
    }

    # create the message with the right data to publish 
    now = rospy.get_rostime()
    while (now.secs == 0 | now.nsecs == 0):
        now = rospy.get_rostime()
    seq = 1
    secs = now.secs
    nsecs = now.nsecs
    message_to_write = FollowJointTrajectoryActionGoal()
    message_to_write.header.stamp.secs = secs
    message_to_write.header.stamp.nsecs = nsecs
    message_to_write.goal_id.stamp.secs = secs
    message_to_write.goal_id.stamp.nsecs = nsecs
    message_to_write.goal.trajectory.joint_names = joint_names
    message_to_write.goal_id.id = "/myWriter-"+str(seq)+"-"+str(secs)+"."+str(nsecs)
    temp_points = []
    trajectory_point = JointTrajectoryPoint()
    trajectory_point.time_from_start.secs = 0
    trajectory_point.time_from_start.nsecs = 0
    for i in range(len(trajectories["joint_trajectory"])):
    # for i in range(int(trajectories["realease_frame"])+1):
        if i != 0:
            trajectory_point.time_from_start.nsecs += dt
            if trajectory_point.time_from_start.nsecs >= 1000000000:
                trajectory_point.time_from_start.secs += int(trajectory_point.time_from_start.nsecs / 1000000000)
                trajectory_point.time_from_start.nsecs = trajectory_point.time_from_start.nsecs % 1000000000
        trajectory_point.positions = trajectories["joint_trajectory"][i]
        trajectory_point.velocities = trajectories["joint_velocity"][i]
        # trajectory_point.accelerations = trajectories["joint_acceleration"][i]
        trajectory_point.accelerations = [0]*len(trajectories["joint_trajectory"][0])
        if i == int(trajectories["realease_frame"]):
            release_time_from_start["secs"] = trajectory_point.time_from_start.secs
            release_time_from_start["nsecs"] = trajectory_point.time_from_start.nsecs
        if i % 5 == 4 or i == 99 or i == 0:
            temp_points.append(copy.deepcopy(trajectory_point))

    for i in range(deceleration_frames):
        trajectory_point.time_from_start.nsecs += deceleration_dt*1000000000
        if trajectory_point.time_from_start.nsecs >= 1000000000:
            trajectory_point.time_from_start.secs += int(trajectory_point.time_from_start.nsecs / 1000000000)
            trajectory_point.time_from_start.nsecs = trajectory_point.time_from_start.nsecs % 1000000000
        trajectory_point.positions = deceleration["joints_positions"][i]
        trajectory_point.velocities = deceleration["joints_velocities"][i]
        trajectory_point.accelerations = [0]*len(trajectories["joint_trajectory"][0])
        # if i % 4 == 0 or i == deceleration_frames-1 or i == 0:
        if i % 1 == 0 or i == deceleration_frames-1 or i == 1:
            temp_points.append(copy.deepcopy(trajectory_point))    
    # trajectory_point.time_from_start.secs = tot_time_nsecs / 1000000000
    # trajectory_point.time_from_start.nsecs += tot_time_nsecs % 1000000000
    # trajectory_point.positions = trajectories["joint_trajectory"][-1]
    # trajectory_point.velocities = [0]*len(trajectories["joint_trajectory"][0])
    # trajectory_point.accelerations = [0]*len(trajectories["joint_trajectory"][0])
    # temp_points.append(copy.deepcopy(trajectory_point))

    # print ("temp_points.positions: ")
    # print (temp_points.positions)

    # for i in range(len(temp_points.positions)):
    #     print ('temp_points.positions[' + str(i) + '] = ' + str(temp_points.positions[i]))

    
    message_to_write.goal.trajectory.points = temp_points
    rospy.loginfo(message_to_write)
    print("=== Press `Enter` to publish ===")
    raw_input()
    pub.publish(message_to_write)

    now = rospy.get_rostime()
    adjusted_release_time["secs"] = now.secs + release_time_from_start["secs"]
    adjusted_release_time["nsecs"] = now.nsecs + release_time_from_start["nsecs"] - gripper_open_delay * 1000000000
    while adjusted_release_time["nsecs"] < 0:
        adjusted_release_time["secs"] -= 1
        adjusted_release_time["nsecs"] += 1000000000
    while adjusted_release_time["nsecs"] >= 1000000000:
        adjusted_release_time["secs"] += 1
        adjusted_release_time["nsecs"] -= 1000000000

    # wait for the release time and then open the gripper
    print("=== Wating to reach the adjusted release time ===")
    print("adjusted_release_time:")
    print("\tadjusted_release_time['secs'] = " + str(adjusted_release_time["secs"]))
    print("\tadjusted_release_time['nsecs'] = " + str(adjusted_release_time["nsecs"]))
    print("release_time_from_start:")
    print("\trelease_time_from_start['secs'] = " + str(release_time_from_start["secs"]))
    print("\trelease_time_from_start['nsecs'] = " + str(release_time_from_start["nsecs"]))
    gripper_move_message = MoveActionGoal()
    gripper_move_message.goal.width = 0.09
    gripper_move_message.goal.speed = 0.05
    
    now = rospy.get_rostime()
    while (now.secs == 0 | now.nsecs == 0):
        now = rospy.get_rostime()
    secs = now.secs
    secs_waited = 0
    while (secs < adjusted_release_time["secs"]):
        secs_waited += 1
        secs = rospy.get_rostime().secs
    while (now.secs <= adjusted_release_time["secs"] and (now.secs < adjusted_release_time["secs"] or now.nsecs < adjusted_release_time["nsecs"])):
        now = rospy.get_rostime()
    gripper_move_pub.publish(gripper_move_message)
    print("secs_waited = " + str(secs_waited))
    print("secs = " + str(secs))
    print("Opening the gripper at:")
    print("\tnow.secs = " + str(now.secs))
    print("\tnow.nsecs = " + str(now.nsecs))
    

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
