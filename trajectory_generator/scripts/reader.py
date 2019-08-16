#!/usr/bin/env python
import os
import sys
import time
import copy
import json
import rospy
import moveit_msgs.msg
import moveit_commander
import geometry_msgs.msg

import argparse

from rospy_message_converter import json_message_converter

from math import pi
from std_msgs.msg import Header
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from actionlib_msgs.msg import GoalStatusArray
from actionlib_msgs.msg import GoalID
from trajectory_msgs.msg import JointTrajectory
from moveit_msgs.msg import MoveGroupActionGoal
from actionlib_msgs.msg import GoalStatusArray
from moveit_msgs.msg import MotionPlanRequest
from control_msgs.msg import JointTrajectoryControllerState
from control_msgs.msg import FollowJointTrajectoryActionGoal
from control_msgs.msg import FollowJointTrajectoryActionFeedback
from moveit_commander.conversions import pose_to_list
from moveit_msgs.msg import ExecuteTrajectoryActionGoal
from moveit_msgs.msg import ExecuteTrajectoryActionResult
from moveit_msgs.msg import ExecuteTrajectoryActionFeedback

import thread

import rospkg
rospack = rospkg.RosPack()

parser = argparse.ArgumentParser(description='rostopic reader')
parser.add_argument('-s', '--sim', nargs='?', const=True, default=False,
                    help='says whether we are in simulation or not')
parser.add_argument('-o', '--output-dir', default=False,
                    help='define the folder where to write the read data')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()

dirName = args.output_dir + str(time.time())
# dirName = args.output_dir + "latest"
# try:
#     dirName = args.o + "latest"
# except(AttributeError):
#     raise AttributeError("Specify an output folder using argument '-o' or '--output-dir'")

os.mkdir(dirName)
dirNameRaw = dirName + "/raw/"
os.mkdir(dirNameRaw)
dirNameJson = dirName + "/json"
os.mkdir(dirNameJson)

def callback(data, args):
    # if args[0] == "/joint_states":
    #     rospy.loginfo(args[0] + "\n%s\n\n------------------\n", data.position)
    
    json_str = json_message_converter.convert_ros_message_to_json(data)
    
    
    json_obj = json.loads(json_str)
    secs = json_obj["header"]["stamp"]["secs"]
    nsecs = json_obj["header"]["stamp"]["nsecs"]
    pos =json_obj["position"][0]
    new_data = (secs, nsecs, pos)
    
    
    if os.stat(args[4]).st_size == 0:
        # args[3].write(json_str)
        args[3].write(json.dumps(new_data)+"\n")
    else:
        # args[3].write(","+json_str)
        args[3].write(json.dumps(new_data)+"\n")
    args[1].write(str(data)+"\n\n------------------------------------\n\n")

    # json_string = json.dumps(str(data))
    # args[1].write(json_string+"\n\n------------------------------------\n\n")


def listener():
    for item in topics_to_read:
        filePath = dirNameRaw+"/"+item[0][1:].replace("/", "-")+".mylog"
        filePathJson = dirNameJson+"/"+item[0][1:].replace("/", "-")+".mylog"
        item[2] = open(filePath, "a+")
        item[3] = open(filePathJson, "a+")
        item[3].write("[")
        rospy.Subscriber(item[0], item[1], callback, [item[0], item[2], filePath, item[3], filePathJson])
        # rospy.Subscriber("/panda_arm_controller/state", JointTrajectoryControllerState, topic2)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

topics_to_read = [
    # [topicName, topicMessage, fileHandler, jsonFileHandler]
    # ["/panda_arm_controller/follow_joint_trajectory/feedback", FollowJointTrajectoryActionFeedback, None, None], #written by /gazebo
    # ["/panda_arm_controller/follow_joint_trajectory/goal", FollowJointTrajectoryActionGoal, None, None], #written by /move_group
    # ["/panda_arm_controller/follow_joint_trajectory/status", GoalStatusArray, None, None], #written by /gazebo
    # ["/panda_arm_controller/state", JointTrajectoryControllerState, None, None], #written by /gazebo
    # ["/panda_arm_controller/command", JointTrajectory, None, None], #written by unknown

    # ["/joint_states", JointState, None, None], #written by /gazebo
    # ["/move_group/goal", MoveGroupActionGoal, None, None], #written by /move_group_commander_wrappers_xxxxxxxxxxxxxxxxx
    # ["/trajectory_execution_event", String, None, None], #written by /move_group_commander_wrappers_xxxxxxxxxxxxxxxxx
    # ["/move_group/status", GoalStatusArray, None, None],
    # ["/move_group/motion_plan_request", MotionPlanRequest, None, None], #written by /move_group
    # ["/execute_trajectory/goal", ExecuteTrajectoryActionGoal, None, None], #written by /move_group_commander_wrappers_xxxxxxxxxxxxxxxxx
    # ["/execute_trajectory/result", ExecuteTrajectoryActionResult, None, None], #written by /move_group_commander_wrappers_xxxxxxxxxxxxxxxxx
    # ["/execute_trajectory/feedback", ExecuteTrajectoryActionFeedback, None, None], #written by /move_group_commander_wrappers_xxxxxxxxxxxxxxxxx
    # ["/execute_trajectory/status", GoalStatusArray, None, None], #written by /move_group_commander_wrappers_xxxxxxxxxxxxxxxxx
    # ["/execute_trajectory/cancel", GoalID, None, None], #written by /move_group_commander_wrappers_xxxxxxxxxxxxxxxxx
    # ["/move_group/fake_controller_joint_states", JointState, None, None], #written by /move_group_commander_wrappers_xxxxxxxxxxxxxxxxx
    # ["/position_joint_trajectory_controller/follow_joint_trajectory/goal", FollowJointTrajectoryActionGoal, None, None], #written by /move_group
    # ["/position_joint_trajectory_controller/follow_joint_trajectory/feedback", FollowJointTrajectoryActionFeedback, None, None], #written by /gazebo
    # ["/position_joint_trajectory_controller/follow_joint_trajectory/status", GoalStatusArray, None, None], #written by /gazebo
    # ["/position_joint_trajectory_controller/state", JointTrajectoryControllerState, None, None], #written by /gazebo
    # ["/position_joint_trajectory_controller/command", JointTrajectory, None, None], #written by unknown
    
    ["/franka_gripper/joint_states", JointState, None, None],
]

if args.sim == False:
    topics_to_read = list(map(lambda item: ["/panda"+item[0], item[1], item[2], item[3]], topics_to_read))

g1 = [-0.0212794437329, -0.620257643868, -0.0216638313722, -2.80111496341, -0.0280020100702, 1.74914135642, -0.0253619484297]
g2 = [0.0292552400818, 0.634437462374, 0.0173031725071, -1.90729899005, 0.0100924668669, 2.11079562192, 0.0381150350331]
g3 = [0.0368132791474, 0.994468192792, 0.0418851799155, -0.980898856146, -0.00708862928778, 2.50000000000, 0.0686074037319]

standing = [7.30342405383e-05, 0.0, -7.93614821509e-05, -0.5, -4.09836759325e-05, 0.5, -1.63242870942e-05]
oldold_start = [0.0, -1.0, 0.0, -2.5, 0.0, 1.5, 0.0]
old_start =     [-0.495593424723, 0.289697317793, -0.578720475955, -1.90095534978, 0.881764211964, 2.25612995392, -1.57274369877]
start =         [-0.448036147657, 0.328661662868, -0.622003205874, -1.82402771276, 0.269721323163, 2.1145116905, -1.94276850845]

testStart = [0.013234692222, -1.33300550194, 1.65682837835, -1.60114211863, 2.82590794707, 1.58192210907, 1.48071221317]
kdl_start = [-1.16812, 0.622, 0.164372, -1.90355, 2.8973, 0.631846, -2.8973]
kdl_start2 = [-1.04314, 0.596338, 0.036396, -1.93809, -2.8973, 0.64496, 2.8973]
kdl_start3 = [0.841403, 0.4, -0.7, -2.22141, -0.25, 1.8, -2.2]
kdl_start4 = [1, 0.4, -0.679573, -1.5, -0.25, 1.8, -1.914]

real_start = [-0.11, 0.06, -0.7, -2.29, 0, 2.3, -1.6]





def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    all_equal = True
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

    return True

class MoveGroupPythonIntefaceTutorial(object):
    """MoveGroupPythonIntefaceTutorial"""
    def __init__(self):
        super(MoveGroupPythonIntefaceTutorial, self).__init__()
        # In ROS, nodes are uniquely named. If two nodes with the same
        # name are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        rospy.init_node('myReader', anonymous=True)
        moveit_commander.roscpp_initialize(sys.argv)

        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()

        group_name = "panda_arm"
        group = moveit_commander.MoveGroupCommander(group_name)

        if args.sim == False:
            display_trajectory_publisher = rospy.Publisher('/panda/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)
        else:
            display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

        # We can get the name of the reference frame for this robot:
        planning_frame = group.get_planning_frame()
        print("============ Reference frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = group.get_end_effector_link()
        print("============ End effector: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print("============ Robot Groups:", robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")

        self.box_name = ''
        self.robot = robot
        self.scene = scene
        self.group = group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

    def go_to_joint_state(self):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        group = self.group

        ## BEGIN_SUB_TUTORIAL plan_to_joint_state
        ##
        ## Planning to a Joint Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^^
        ## The Panda's zero configuration is at a `singularity <https://www.quora.com/Robotics-What-is-meant-by-kinematic-singularity>`_ so the first
        ## thing we want to do is move it to a slightly better configuration.
        # We can get the joint values from the group and adjust some of the values:
        joint_goal = group.get_current_joint_values()
        # for i in range(len(g1)):
        #     joint_goal[i] = g1[i]
        for i in range(len(start)):
            joint_goal[i] = real_start[i]
            # joint_goal[i] = kdl_start4[i]
            # joint_goal[i] = start[i]
            # joint_goal[i] = testStart[i]

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        group.stop()

        ## END_SUB_TUTORIAL

        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_joints = self.group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def plan_cartesian_path(self, scale=1):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        group = self.group

        ## BEGIN_SUB_TUTORIAL plan_cartesian_path
        ##
        ## Cartesian Paths
        ## ^^^^^^^^^^^^^^^
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through:
        ##
        waypoints = []

        s = group.get_current_pose().pose
        
        # print("current_pose:")
        # print(s)
        
        wpose = s
        print("waypoint 0: ", wpose)

        wpose.position.z -= scale * 0.1
        wpose.position.x += scale * 0.1
        waypoints.append(copy.deepcopy(wpose))
        print("waypoint 1: ", wpose)

        # wpose.position.x += scale * 0.5
        # waypoints.append(copy.deepcopy(wpose))
        # print("waypoint 2: ", wpose)

        # wpose.position.z += scale * 0.1
        # wpose.position.x += scale * 0.1
        # waypoints.append(copy.deepcopy(wpose))
        # print("waypoint 3: ", wpose)

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0 disabling:
        (plan, fraction) = group.compute_cartesian_path(
                                        waypoints,   # waypoints to follow
                                        0.01,        # eef_step
                                        0.0)         # jump_threshold

        # Note: We are just planning, not asking move_group to actually move the robot yet:
        return plan, fraction

        ## END_SUB_TUTORIAL

    def display_trajectory(self, plan):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        robot = self.robot
        display_trajectory_publisher = self.display_trajectory_publisher

        ## BEGIN_SUB_TUTORIAL display_trajectory
        ##
        ## Displaying a Trajectory
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## You can ask RViz to visualize a plan (aka trajectory) for you. But the
        ## group.plan() method does this automatically so this is not that useful
        ## here (it just displays the same trajectory again):
        ##
        ## A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        ## We populate the trajectory_start with our current robot state to copy over
        ## any AttachedCollisionObjects and add our plan to the trajectory.
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        display_trajectory_publisher.publish(display_trajectory);

        ## END_SUB_TUTORIAL

    def execute_plan(self, plan):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        group = self.group

        ## BEGIN_SUB_TUTORIAL execute_plan
        ##
        ## Executing a Plan
        ## ^^^^^^^^^^^^^^^^
        ## Use execute if you would like the robot to follow
        ## the plan that has already been computed:
        group.execute(plan, wait=True)

        ## **Note:** The robot's current joint state must be within some tolerance of the
        ## first waypoint in the `RobotTrajectory`_ or ``execute()`` will fail
        ## END_SUB_TUTORIAL

def main():
    try:
        print("============ Press `Enter` to begin (press ctrl-d to exit) ...")
        raw_input()
        tutorial = MoveGroupPythonIntefaceTutorial()

        thread.start_new_thread(listener, ())
        # listener()
        print("============ Press `Enter` to execute a movement using a joint state goal ...")
        raw_input()
        tutorial.go_to_joint_state()

        # print("============ Press `Enter` to execute a movement using a pose goal ...")
        # raw_input()
        # tutorial.go_to_pose_goal()

        print("============ Press `Enter` to plan and display a Cartesian path ...")
        raw_input()
        cartesian_plan, fraction = tutorial.plan_cartesian_path()

        print("============ Press `Enter` to display a saved trajectory (this will replay the Cartesian path)  ...")
        raw_input()
        tutorial.display_trajectory(cartesian_plan)

        print("============ Press `Enter` to execute a saved path ...")
        raw_input()
        tutorial.execute_plan(cartesian_plan)

        # print("============ Press `Enter` to add a box to the planning scene ...")
        # raw_input()
        # tutorial.add_box()

        # print("============ Press `Enter` to attach a Box to the Panda robot ...")
        # raw_input()
        # tutorial.attach_box()

        # print("============ Press `Enter` to plan and execute a path with an attached collision object ...")
        # raw_input()
        # cartesian_plan, fraction = tutorial.plan_cartesian_path(scale=-1)
        # tutorial.execute_plan(cartesian_plan)

        # print("============ Press `Enter` to detach the box from the Panda robot ...")
        # raw_input()
        # tutorial.detach_box()

        # print("============ Press `Enter` to remove the box from the planning scene ...")
        # raw_input()
        # tutorial.remove_box()

        time.sleep(2)
        for item in topics_to_read:
            item[2].close()
            item[3].write("]")
            item[3].close()
        print("============ complete!")
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return

if __name__ == '__main__':
    main()