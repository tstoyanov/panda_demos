#include "panda_insertion/MessageHandler.hpp"

#include <eigen_conversions/eigen_msg.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf/tf.h>

#include <cmath>

using namespace std;

MessageHandler::MessageHandler() {}

MessageHandler::MessageHandler(ros::NodeHandle* nodeHandler, Panda* panda)
{
    this->nodeHandler = nodeHandler;
    this->panda = panda;
    this->baseFrameId = "panda_link0";
}

PoseStampedMsg MessageHandler::pointPoseMessage(Point point)
{
    PoseStampedMsg message = emptyPoseMessage();

    message.header.frame_id = baseFrameId;
   // ROS_DEBUG_STREAM("baseFrameId: " << baseFrameId);
    
    message.pose.position.x = point.x;
    message.pose.position.y = point.y;
    message.pose.position.z = point.z;

    message.pose.orientation = panda->straightOrientation;
    
    return message;
}

double MessageHandler::sineYaw(PoseStampedMsg& message, int i)
{    
    // get RPY
    mutex.lock();
    geometry_msgs::Transform transform = panda->transformStamped.transform;
    mutex.unlock();
    tf::Quaternion myQuaternion(transform.rotation.x, 
                                transform.rotation.y,
                                transform.rotation.z,
                                transform.rotation.w);
    tf::Matrix3x3 m(myQuaternion);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
                                                         
    // sine Yaw
    float alpha = 0.4;
    yaw = alpha*sin(0.05*i);
    //std::cout << "++sine yaw:" << yaw << std::endl;
    myQuaternion.setRPY(roll, pitch, yaw);
                                                                                              
    message.pose.orientation.x = myQuaternion.x();
    message.pose.orientation.y = myQuaternion.y();
    message.pose.orientation.z = myQuaternion.z();
    message.pose.orientation.w = myQuaternion.w();

    return yaw;
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

PoseStampedMsg MessageHandler::insertionWigglePoseMessage(double xAng, double yAng)
{
    geometry_msgs::PoseStamped message = emptyPoseMessage();
    message.header.frame_id = "panda_link0";

    // Get goal parameter from server
    vector<double> goal;
    const string goalParameter = "/wiggle/goal";
    if (!nodeHandler->getParam(goalParameter, goal))
    {
        throw runtime_error("Could not get parameter from server");
    }

    mutex.lock();
    const geometry_msgs::Transform transMsg = panda->transformStamped.transform;
    mutex.unlock();

    // Read robot pose
    message.pose.position.x = goal.at(0);
    message.pose.position.y = goal.at(1);
    message.pose.position.z = goal.at(2);
    
    message.pose.orientation.x = transMsg.rotation.x;
    message.pose.orientation.y = transMsg.rotation.y;
    message.pose.orientation.z = transMsg.rotation.z;
    message.pose.orientation.w = transMsg.rotation.w;

    // Convert to matrix
    Eigen::Affine3d tMatrix;
    tf::poseMsgToEigen(message.pose, tMatrix);

    double roll = (xAng * 1.01 * M_PI);
    double pitch = (yAng * 1.05 * M_PI);
    double yaw = (0.0 * M_PI);

    Eigen::Affine3d rotated_tMatrix = rotateMatrixRPY(tMatrix, roll, pitch, yaw);

    //ROS_DEBUG_STREAM("roll: " << roll << ", pitch: " << pitch << ", yaw: " << yaw);
    //ROS_DEBUG_STREAM("rotated_tMatrix: " << endl << rotated_tMatrix.rotation());

    // Convert back to msg
    tf::poseEigenToMsg(rotated_tMatrix, message.pose);

    return message;
}

PoseStampedMsg MessageHandler::straighteningPoseMessage()
{
    geometry_msgs::PoseStamped message = emptyPoseMessage();
    message.header.frame_id = baseFrameId;

    // Get goal parameter from server
    vector<double> goal;
    const string goalParameter = "/straightening/goal";
    if (!nodeHandler->getParam(goalParameter  , goal))
    {
        throw runtime_error("Could not get parameter from server");
    }

    mutex.lock();
    const geometry_msgs::Transform transMsg = panda->transformStamped.transform;
    mutex.unlock();

    // Read robot pose
    message.pose.position.x = transMsg.translation.x;
    message.pose.position.y = transMsg.translation.y;
    message.pose.position.z = goal.at(2);

    message.pose.orientation = panda->straightOrientation;

    return message;
}

Eigen::Affine3d MessageHandler::rotateMatrixRPY(Eigen::Affine3d tMatrix, double rollAngle,
                                                double pitchAngle, double yawAngle)
{
    Eigen::AngleAxisd roll(rollAngle, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch(pitchAngle, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw(yawAngle, Eigen::Vector3d::UnitZ());

//    Eigen::Quaterniond quaternion = roll * pitch * yaw;

//    Eigen::Affine3d rotated_tMatrix = tMatrix.rotate(quaternion);
    Eigen::Affine3d rotated_tMatrix = tMatrix*(roll*pitch*yaw);

    return rotated_tMatrix;
}

geometry_msgs::Pose MessageHandler::generateRobotPoseMessage()
{
    geometry_msgs::Pose poseMsg;

    mutex.lock();
    const geometry_msgs::Transform transMsg = panda->transformStamped.transform;
    mutex.unlock();

    // Read robot pose
    poseMsg.position.x = transMsg.translation.x;
    poseMsg.position.y = transMsg.translation.y;
    poseMsg.position.z = transMsg.translation.z;
    
    poseMsg.orientation.x = transMsg.rotation.x;
    poseMsg.orientation.y = transMsg.rotation.y;
    poseMsg.orientation.z = transMsg.rotation.z;
    poseMsg.orientation.w = transMsg.rotation.w;

    return poseMsg;
}

geometry_msgs::Pose MessageHandler::generateRobotErrorPoseMessage()
{
    geometry_msgs::Pose poseMsg;

    // Read robot pose
    mutex.lock();
    const geometry_msgs::Transform transMsg = panda->transformStamped.transform;
    mutex.unlock();

    // Apply translational error
    poseMsg.position.x = transMsg.translation.x * 1.3;
    poseMsg.position.y = transMsg.translation.y * 1.3;
    poseMsg.position.z = transMsg.translation.z * 1.3;

    // Apply rotation error
    poseMsg.orientation.x = transMsg.rotation.x;
    poseMsg.orientation.y = transMsg.rotation.y;
    poseMsg.orientation.z = transMsg.rotation.z;
    poseMsg.orientation.w = transMsg.rotation.w;

    Eigen::Affine3d tMatrix;
    tf::poseMsgToEigen(poseMsg, tMatrix);

    double roll = (0.25 * M_PI);
    double pitch = (0.25 * M_PI);
    double yaw = (0.25 * M_PI);

    Eigen::Affine3d rotated_tMatrix = rotateMatrixRPY(tMatrix, roll, pitch, yaw);

    // Convert back to msg
    tf::poseEigenToMsg(rotated_tMatrix, poseMsg);

    return poseMsg;
}
