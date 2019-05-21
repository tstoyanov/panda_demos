#include "panda_insertion/MessageHandler.hpp"

#include <eigen_conversions/eigen_msg.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <cmath>

using namespace std;

MessageHandler::MessageHandler() {}

MessageHandler::MessageHandler(ros::NodeHandle* nodeHandler, Panda* panda)
{
    this->nodeHandler = nodeHandler;
    this->panda = panda;
}

PoseStampedMsg MessageHandler::pointPoseMessage(Point point)
{
    PoseStampedMsg message = emptyPoseMessage();

    message.header.frame_id = baseFrameId;
    
    message.pose.position.x = point.x;
    message.pose.position.y = point.y;
    message.pose.position.z = point.z;

    // message.pose.orientation = panda->initialOrientation;
    message.pose.orientation.x = 0.983;
    message.pose.orientation.y = 0.186;
    message.pose.orientation.z = 0.002;
    message.pose.orientation.w = 0.001;

    return message;
}

JointTrajectoryMsg MessageHandler::initialJointTrajectoryMessage()
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
    //vector<double> initialPose {0.35, -0.07, -0.23, -2.35, -0.12, 2.28, 0.75};
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

PoseStampedMsg MessageHandler::downMovementPoseMessage(double zCoordinate)
{
    geometry_msgs::PoseStamped message = emptyPoseMessage();
    message.pose.position = panda->initialPosition;

    panda->position.z = zCoordinate;

    ROS_DEBUG_STREAM("Panda position z: " << panda->position.z);

    message.header.frame_id = baseFrameId;
    message.pose.position = panda->position;
    message.pose.orientation = panda->orientation;
    
    return message;
}

PoseStampedMsg MessageHandler::emptyPoseMessage()
{
    geometry_msgs::PoseStamped message;

    // Header
    string frameId = "";
    ros::Time stamp(0.0);
    uint32_t seq = 0;

    // Points
    double position_x = 0.0;
    double position_y = 0.0;
    double position_z = 0.0;

    double orientation_x = 0.0;
    double orientation_y = 0.0;
    double orientation_z = 0.0;
    double orientation_w = 0.0;

    message.header.frame_id = frameId;
    message.header.stamp = stamp;
    message.header.seq = seq;

    message.pose.position.x = position_x;
    message.pose.position.y = position_y;
    message.pose.position.z = position_z;

    message.pose.orientation.x = orientation_x;
    message.pose.orientation.y = orientation_y;
    message.pose.orientation.z = orientation_z;
    message.pose.orientation.w = orientation_w;

    return message;
}

PoseStampedMsg MessageHandler::spiralPointPoseMessage(Point point)
{
    geometry_msgs::PoseStamped message = emptyPoseMessage();

    message.header.frame_id = baseFrameId;
    message.pose.position.x = point.x;
    message.pose.position.y = point.y;
    message.pose.position.z = point.z;

    message.pose.orientation = panda->orientation;

    return message;
}

PoseStampedMsg MessageHandler::insertionWigglePoseMessage(double xAngle)
{
    geometry_msgs::PoseStamped message = emptyPoseMessage();
    message.pose.orientation = panda->orientation;
    message.pose.position = panda->position;

    // Convert to matrix
    Eigen::Affine3d tMatrix;
    tf::poseMsgToEigen(message.pose, tMatrix);

    double roll = (xAngle * M_PI);
    double pitch = (0.0 * M_PI);
    double yaw = (0.0 * M_PI);
    Eigen::Affine3d rotated_tMatrix = rotateMatrixRPY(tMatrix, roll, pitch, yaw);
    ROS_DEBUG_STREAM("rotated_tMatrix: " << endl << rotated_tMatrix.rotation());

    // Convert back to msg
    tf::poseEigenToMsg(rotated_tMatrix, message.pose);

    // Set new orientation
    message.header.frame_id = baseFrameId;
    panda->orientation = message.pose.orientation;

    return message;
}

PoseStampedMsg MessageHandler::straighteningPoseMessage()
{
    geometry_msgs::PoseStamped message = emptyPoseMessage();

    message.header.frame_id = baseFrameId;
    message.pose.position = panda->position;
    message.pose.orientation = panda->orientation;
    message.pose.orientation.x = 1;
    message.pose.orientation.y = 0;
    message.pose.orientation.z = 0;
    message.pose.orientation.w = 0;
    panda->orientation = message.pose.orientation;

    return message;
}

Eigen::Affine3d MessageHandler::rotateMatrixRPY(Eigen::Affine3d tMatrix, double rollAngle,
                                                double pitchAngle, double yawAngle)
{
    Eigen::AngleAxisd roll(rollAngle, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch(pitchAngle, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw(yawAngle, Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond quaternion = yaw * pitch * roll;

    Eigen::Affine3d rotated_tMatrix = tMatrix.rotate(quaternion);

    return rotated_tMatrix;
}

