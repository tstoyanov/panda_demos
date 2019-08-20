#include "ros/ros.h"
#include "trajectory_generator/trajectory_generator_gripper.h"

#include <franka_gripper/GraspAction.h>
#include <franka_gripper/HomingAction.h>
#include <franka_gripper/MoveAction.h>
#include <franka_gripper/StopAction.h>
#include <franka_gripper/franka_gripper.h>

#include <franka/gripper_state.h>

#include <cmath>
#include <functional>
#include <thread>
#include <mutex>
#include <string>
#include <vector>

#include <ros/init.h>
#include <ros/node_handle.h>
#include <ros/rate.h>
#include <ros/spinner.h>
#include <sensor_msgs/JointState.h>


namespace
{

template <typename T_action, typename T_goal, typename T_result>
void handleErrors(actionlib::SimpleActionServer<T_action> *server,
                  std::function<bool(const T_goal &)> handler,
                  const T_goal &goal)
{
    T_result result;
    try
    {
        result.success = handler(goal);
        server->setSucceeded(result);
    }
    catch (const franka::Exception &ex)
    {
        ROS_ERROR_STREAM("" << ex.what());
        result.success = false;
        result.error = ex.what();
        server->setAborted(result);
    }
}

} // namespace

// using actionlib::SimpleActionServer;
using control_msgs::GripperCommandAction;
using franka_gripper::grasp;
using franka_gripper::GraspAction;
using franka_gripper::GraspEpsilon;
using franka_gripper::GraspGoalConstPtr;
using franka_gripper::GraspResult;
using franka_gripper::gripperCommandExecuteCallback;
using franka_gripper::homing;
using franka_gripper::HomingAction;
using franka_gripper::HomingGoalConstPtr;
using franka_gripper::HomingResult;
using franka_gripper::move;
using franka_gripper::MoveAction;
using franka_gripper::MoveGoalConstPtr;
using franka_gripper::MoveResult;
using franka_gripper::stop;
using franka_gripper::StopAction;
using franka_gripper::StopGoalConstPtr;
using franka_gripper::StopResult;
using franka_gripper::updateGripperState;


bool add(trajectory_generator::trajectory_generator_gripper::Request &req,
         trajectory_generator::trajectory_generator_gripper::Response &res)
{
    res.sum = req.a + req.b;
    ROS_INFO("request: x=%ld, y=%ld", (long int)req.a, (long int)req.b);
    ROS_INFO("sending back response: [%ld]", (long int)res.sum);
    return true;
}

bool my_move(const franka::Gripper& gripper, const MoveGoalConstPtr& goal, trajectory_generator::trajectory_generator_gripper::Response &res) {
    res.sum = 37;
    ROS_INFO("request received");
    return true;
//   return gripper.move(goal->width, goal->speed);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "trajectory_generator_gripper_server");
    ros::NodeHandle node_handle;
    std::string robot_ip;
    if (!node_handle.getParam("robot_ip", robot_ip))
    {
        ROS_ERROR("franka_gripper_node: Could not parse robot_ip parameter");
        return -1;
    }

    double default_speed(0.1);
    if (node_handle.getParam("default_speed", default_speed))
    {
        ROS_INFO_STREAM("franka_gripper_node: Found default_speed " << default_speed);
    }

    GraspEpsilon default_grasp_epsilon;
    default_grasp_epsilon.inner = 0.005;
    default_grasp_epsilon.outer = 0.005;
    std::map<std::string, double> epsilon_map;
    if (node_handle.getParam("default_grasp_epsilon", epsilon_map))
    {
        ROS_INFO_STREAM("franka_gripper_node: Found default_grasp_epsilon "
                        << "inner: " << epsilon_map["inner"] << ", outer: " << epsilon_map["outer"]);
        default_grasp_epsilon.inner = epsilon_map["inner"];
        default_grasp_epsilon.outer = epsilon_map["outer"];
    }

    franka::Gripper gripper(robot_ip);

    auto homing_handler = [&gripper](auto &&goal) { return homing(gripper, goal); };
    auto stop_handler = [&gripper](auto &&goal) { return stop(gripper, goal); };
    auto grasp_handler = [&gripper](auto &&goal) { return grasp(gripper, goal); };
    auto move_handler = [&gripper](auto &&goal) { return move(gripper, goal); };

    ros::ServiceServer service = node_handle.advertiseService("trajectory_generator_gripper", move);
    // ros::ServiceServer service = node_handle.advertiseService("trajectory_generator_gripper", add);
    ROS_INFO("Ready to add two ints.");
    ros::spin();

    return 0;
}