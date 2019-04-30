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
import json, ast, collections, sys, getopt

from rospy_message_converter import json_message_converter

import thread

folder_name = "test"
# if len(sys.arvg) == 2:
#     folder_name = sys.argv[1]

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
        folder_name = arg

print ("folder_name: " + folder_name)

# with open("/home/ilbetzy/orebro/trajectory_generation_ws/generated_trajectories/cpp/" + folder_name + "/trajectories.txt", 'r') as f:
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

start = [-0.448036147657, 0.328661662868, -0.622003205874, -1.82402771276, 0.269721323163, 2.1145116905, -1.94276850845]

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
            # joint_goal[i] = kdl_start4[i]
            joint_goal[i] = start[i]
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

        with open("/home/ilbetzy/orebro/trajectory_generation_ws/generated_trajectories/cpp/" + folder_name + "/trajectories.txt", 'r') as f:
            data = f.read()
        trajectories = json.loads(data)
        trajectories = ast.literal_eval(json.dumps(trajectories))

        joint_trajectories = {}
        for i in range(len(trajectories["joint_trajectory"][0])):
            joint_trajectories[i] = []
        for values in trajectories["joint_trajectory"]:
            for i in range(len(values)):
                joint_trajectories[i].append(values[i])

        eef_pose = {
            "x": [],
            "y": [],
            "z": []
        }
        for values in trajectories["eef_trajectory"]:
            eef_pose["x"].append(values["x"])
            eef_pose["y"].append(values["y"])
            eef_pose["z"].append(values["z"])

        for i in range(len(eef_pose["x"])):
            wpose.position.x = eef_pose["x"][i]
            wpose.position.y = eef_pose["y"][i]
            wpose.position.z = eef_pose["z"][i]
        
            waypoints.append(copy.deepcopy(wpose))

        # wpose.position.z -= scale * 0.2
        # wpose.position.x += scale * 0.4
        # waypoints.append(copy.deepcopy(wpose))
        # print("waypoint 1: ", wpose)

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

        print ("plan: ", plan)

        # ====================JSON====================
        
        dirName = "/home/ilbetzy/orebro/trajectory_generation_ws/generated_trajectories/moveit/" + str(time.time())
        os.mkdir(dirName)

        filePath = dirName+"/trajectories.txt"
        fileHandler = open(filePath, "a+")

        json_str = json_message_converter.convert_ros_message_to_json(plan)
        joints_state = plan.joint_trajectory.points[0].positions

        # ========== FK SERVICE ==========
        # rospy.wait_for_service('compute_fk')
        # self.compute_fk = rospy.ServiceProxy('compute_fk', GetPositionFK)
        # eef_pose = compute_fk_client(group, joints_state, ['panda_hand'])
        # ========== FK SERVICE ==========

        if os.stat(filePath).st_size == 0:
            fileHandler.write(json_str)
        else:
            fileHandler.write(","+json_str)

        # json data;
        # for (unsigned i = 0; i < joint_trajectory.size(); i++)
        # {
        #     for (unsigned ii = 0; ii < nr_of_joints; ii++)
        #     {
        #     data["joint_trajectory"][i][ii] = joint_trajectory[i].data[ii];
        #     }

        #     data["eef_trajectory"][i]["x"] = eef_trajectory[i].x();
        #     data["eef_trajectory"][i]["y"] = eef_trajectory[i].y();
        #     data["eef_trajectory"][i]["z"] = eef_trajectory[i].z();

        #     data["fk_eef_trajectory"][i]["x"] = fk_eef_trajectory[i].x();
        #     data["fk_eef_trajectory"][i]["y"] = fk_eef_trajectory[i].y();
        #     data["fk_eef_trajectory"][i]["z"] = fk_eef_trajectory[i].z();
        # }
        # // std::cout << "\n\njson:\n" << data << std::endl;

        # long long ts = std::chrono::system_clock::now().time_since_epoch().count();
        # std::cout << "time: " << ts << std::endl;
        # std::string dir_path = "./generated_trajectories/" + std::to_string(ts);
        # boost::filesystem::path dir(dir_path);
        # if(!(boost::filesystem::exists(dir)))
        # {
        #     if (boost::filesystem::create_directory(dir))
        #         std::cout << "....Folder Successfully Created!" << std::endl;
        # }
        # std::ofstream myfile (dir_path + "/trajectories.txt");
        # if (myfile.is_open())
        # {
        #     myfile << data << std::endl;
        #     myfile.close();
        # }
        # ====================END JSON====================

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
        print("============ Press `Enter` to begin the tutorial by setting up the moveit_commander (press ctrl-d to exit) ...")
        raw_input()
        tutorial = MoveGroupPythonIntefaceTutorial()

        # thread.start_new_thread(listener, ())
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

        print("============ END ============")
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return

if __name__ == '__main__':
    main()