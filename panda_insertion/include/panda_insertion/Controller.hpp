#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "ros/ros.h"
#include "string"
#include "geometry_msgs/PoseStamped.h"
#include "trajectory_msgs/JointTrajectory.h"
#include "ros/duration.h"
#include "panda_insertion/Panda.hpp"
#include <Eigen/Geometry>


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


typedef struct Point
{
    double x, y, z;
} Point;

typedef struct OrientationRPY
{
    double r, p, y;
} OrientationRPY;

typedef std::vector<Point> Trajectory;

class Controller
{
private:
    ros::NodeHandle* nodeHandler;
    Panda* panda;

    ros::Publisher jointTrajectoryPublisher;
    ros::Publisher equilibriumPosePublisher;
    double loop_rate;

public:
    // Constructor
    Controller();

    // Public methods
    void init(ros::NodeHandle*, Panda* panda);

    void startState();
    bool initialPositionState();
    bool moveToInitialPositionState();
    bool externalDownMovementState();
    bool spiralMotionState();
    bool insertionWiggleState();
    bool straighteningState();
    bool internalDownMovementState();

private:
    void initJointTrajectoryPublisher();
    void initEquilibriumPosePublisher();

    bool loadController(std::string controller);
    bool switchController(std::string from, std::string to);

    geometry_msgs::PoseStamped initialPoseMessage();
    trajectory_msgs::JointTrajectory initialJointTrajectoryMessage();
    geometry_msgs::PoseStamped downMovementPoseMessage(double z_coord);
    geometry_msgs::PoseStamped emptyPoseMessage();
    geometry_msgs::PoseStamped spiralPointPoseMessage(Point point);
    geometry_msgs::PoseStamped insertionWigglePoseMessage(double x_angle);
    geometry_msgs::PoseStamped straighteningPoseMessage();

    void setParameterStiffness(Stiffness stiffness);
    void setParameterDamping(Damping damping);

    Trajectory generateArchimedeanSpiral(double a, double b, int nrOfPoints);

    void writeTrajectoryToFile(Trajectory trajectory, const std::string& fileName, bool appendToFile = false);

    Eigen::Affine3d rotateMatrixRPY(Eigen::Affine3d tMatrix, double rollAngle, double pitchAngle, double yawAngle);
};

#endif