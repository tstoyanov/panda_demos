#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "ros/ros.h"
#include "string"
#include "geometry_msgs/PoseStamped.h"
#include "trajectory_msgs/JointTrajectory.h"
#include "ros/duration.h"

class Controller
{
private:
    ros::NodeHandle* nodeHandler;
    ros::Publisher jointTrajectoryPublisher;
    ros::Publisher equilibriumPosePublisher;
    double loop_rate;

public:
    // Constructors
    Controller();

    // Accessors


    // Manipulators 


    // Public methods
    void init(ros::NodeHandle*);

    void startState();
    bool initialPositionState();
    bool moveToInitialPositionState();
    bool externalDownMovementState();

private:
    void initJointTrajectoryPublisher();
    void initEquilibriumPosePublisher();

    bool loadController(std::string controller);
    bool switchController(std::string from, std::string to);

    geometry_msgs::PoseStamped initialPoseMessage();
    trajectory_msgs::JointTrajectory initialJointTrajectoryMessage();
};

#endif