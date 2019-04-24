#include "panda_insertion/Controller.hpp"
#include "geometry_msgs/PoseStamped.h"
#include "trajectory_msgs/JointTrajectory.h"
#include "ros/console.h"
#include "ros/duration.h"
#include "ros/time.h"
#include "stdint.h"
#include "string"
#include "vector"
#include <iostream>

using namespace std;

Controller::Controller()
{
    ROS_DEBUG("Initialize publishers");
    initEquilibriumPosePublisher();
    initJointTrajectoryPublisher();
}

void Controller::startState()
{
    ROS_DEBUG("Start of start state");
    ros::Duration(2.0).sleep();

    trajectory_msgs::JointTrajectory initialPoseMessage = initialJointTrajectoryMessage();
    jointTrajectoryPublisher.publish(initialPoseMessage);
}

bool Controller::initialPositionState()
{
    ROS_DEBUG_ONCE("Initial position from controller");
}

void Controller::initJointTrajectoryPublisher()
{
    const string topic = "position_joint_trajectory_controller/command";
    const int queueSize = 1000;

    jointTrajectoryPublisher = nodeHandler.advertise<trajectory_msgs::JointTrajectory>(topic, queueSize);
}

void Controller::initEquilibriumPosePublisher()
{
    const string topic = "impedance_controller/equilibrium_pose";
    const int queueSize = 1000;

    equilibriumPosePublisher = nodeHandler.advertise<geometry_msgs::PoseStamped>(topic, queueSize);
}

trajectory_msgs::JointTrajectory Controller::initialJointTrajectoryMessage()
{
    trajectory_msgs::JointTrajectory message;

    // Header
    string frameId = "";
    ros::Time stamp(0.0);
    uint32_t seq = 0;

    // Joint names
    vector<string> jointNames = {"panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                                 "panda_joint5", "panda_joint6", "panda_joint7"};
    ros::Duration timeFromStart(3.0);

    // Points
    vector<double> initialPose {0.81, -0.78, -0.17, -2.35, -0.12, 1.60, 0.75};
    vector<double> effort {0, 0, 0, 0, 0, 0, 0};
    vector<double> accelerations {0, 0, 0, 0, 0, 0, 0};
    vector<double> velocities {0, 0, 0, 0, 0, 0, 0};

    message.points.resize(1);
    message.points[0].effort.resize(7);
    message.points[0].accelerations.resize(7);
    message.points[0].velocities.resize(7);
    message.points[0].positions.resize(7);

    message.header.frame_id = "";
    message.header.stamp = stamp;
    message.header.seq = seq;
    message.joint_names = jointNames;
    message.points[0].positions = initialPose;
    message.points[0].velocities = velocities;
    message.points[0].effort = effort;
    message.points[0].accelerations = accelerations;
    message.points[0].time_from_start = timeFromStart;

    return message;
}