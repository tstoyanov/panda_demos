#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "ros/ros.h"
#include "string"
#include "geometry_msgs/PoseStamped.h"
#include "trajectory_msgs/JointTrajectory.h"
#include "ros/duration.h"
#include "panda_insertion/Panda.hpp"

typedef struct Stiffness
{
    int translational_x;
    int translational_y;
    int translational_z;
    int rotational_x;
    int rotational_y;
    int rotational_z;
} Stiffness;

typedef struct Damping
{
    int translational_x;
    int translational_y;
    int translational_z;
    int rotational_x;
    int rotational_y;
    int rotational_z;
} Damping;

class Controller
{
private:
    ros::NodeHandle* nodeHandler;
    Panda* panda;

    ros::Publisher jointTrajectoryPublisher;
    ros::Publisher equilibriumPosePublisher;
    double loop_rate;

public:
    // Constructors
    Controller();

    // Accessors


    // Manipulators 


    // Public methods
    void init(ros::NodeHandle*, Panda* panda);

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
    geometry_msgs::PoseStamped externalDownMovementPoseMessage();
    geometry_msgs::PoseStamped emptyPoseMessage();

    void setParameterStiffness(Stiffness stiffness);
    void setParameterDamping(Damping damping);
};

#endif