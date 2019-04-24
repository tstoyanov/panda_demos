#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "ros/ros.h"
#include "trajectory_msgs/JointTrajectory.h"

class Controller
{
private:
    ros::NodeHandle nodeHandler;
    ros::Publisher jointTrajectoryPublisher;
    ros::Publisher equilibriumPosePublisher;

public:
    Controller();
    void startState();
    bool initialPositionState();
    void initJointTrajectoryPublisher();
    void initEquilibriumPosePublisher();
    trajectory_msgs::JointTrajectory initialJointTrajectoryMessage();
};

#endif