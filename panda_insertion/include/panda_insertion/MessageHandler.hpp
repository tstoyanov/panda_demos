#ifndef MESSAGE_HANDLER_H
#define MESSAGE_HANDLER_H

#include "ros/ros.h"

#include "panda_insertion/TrajectoryHandler.hpp"
#include "panda_insertion/Panda.hpp"

#include "geometry_msgs/PoseStamped.h"
#include "trajectory_msgs/JointTrajectory.h"

#include <Eigen/Geometry>

typedef geometry_msgs::PoseStamped PoseStampedMsg;
typedef trajectory_msgs::JointTrajectory JointTrajectoryMsg;

class MessageHandler
{
private:
    ros::NodeHandle* nodeHandler;
    Panda* panda;

    std::string baseFrameId;

public:
    MessageHandler();
    MessageHandler(ros::NodeHandle* nodeHandler, Panda* panda);

    PoseStampedMsg pointPoseMessage(Point point);
    JointTrajectoryMsg initialJointTrajectoryMessage();
    PoseStampedMsg downMovementPoseMessage(double zCoordinate);
    PoseStampedMsg emptyPoseMessage();
    PoseStampedMsg spiralPointPoseMessage(Point point);
    PoseStampedMsg insertionWigglePoseMessage(double xAngle);
    PoseStampedMsg straighteningPoseMessage();

    Eigen::Affine3d rotateMatrixRPY(Eigen::Affine3d tMatrix, double rollAngle,
                                    double pitchAngle, double yawAngle);
};

#endif
