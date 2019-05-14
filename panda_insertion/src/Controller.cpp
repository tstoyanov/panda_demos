#include "panda_insertion/Controller.hpp"
#include "panda_insertion/Panda.hpp"

#include <iostream>
#include <fstream>
#include "stdint.h"
#include "string"
#include "vector"

#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Core>
#include <eigen_conversions/eigen_msg.h>

#include "geometry_msgs/PoseStamped.h"
#include "trajectory_msgs/JointTrajectory.h"
#include "controller_manager_msgs/LoadController.h"
#include "controller_manager_msgs/SwitchController.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include "ros/console.h"
#include "ros/duration.h"
#include "ros/package.h"
#include "ros/time.h"

using namespace std;

// Constructors
Controller::Controller() {}

// Public methods
void Controller::init(ros::NodeHandle* nodeHandler, Panda* panda)
{
    loop_rate = 1;
    this->nodeHandler = nodeHandler;
    this->panda = panda;

    initEquilibriumPosePublisher();
    initJointTrajectoryPublisher();
}

void Controller::startState()
{
    trajectory_msgs::JointTrajectory initialPoseMessageJoint = initialJointTrajectoryMessage();
    jointTrajectoryPublisher.publish(initialPoseMessageJoint);
}

bool Controller::moveToInitialPositionState()
{
    string fromController;
    nodeHandler->getParam("insertion/positionJointTrajectoryController", fromController);

    string toController;
    nodeHandler->getParam("insertion/impedanceController", toController);

    ROS_DEBUG_STREAM("fromController:" << fromController );
    ROS_DEBUG_STREAM("toController:" << toController );
    
    if (!loadController(toController))
    {
        return false;
    }

    switchController(fromController, toController);

    geometry_msgs::PoseStamped initialPositionMessage = initialPoseMessage();

    int i = 0;
    ros::Rate rate(loop_rate);
    while (ros::ok() && i < 130)
    {
        equilibriumPosePublisher.publish(initialPositionMessage);
        rate.sleep();
        i++;
    }

    return true;
}

bool Controller::externalDownMovementState()
{
    ROS_DEBUG_ONCE("In external down movement state from controller");

    Stiffness stiffness;
    Damping damping;
    stiffness.translational_x = 5000;
    stiffness.translational_y = 5000;
    stiffness.translational_z = 5000;
    stiffness.rotational_x = 5000;
    stiffness.rotational_y = 5000;
    stiffness.rotational_z = 5000;

    damping.translational_x = 65;
    damping.translational_y = 65;
    damping.translational_z = 65;
    damping.rotational_x = 45;
    damping.rotational_y = 45;
    damping.rotational_z = 45;

    // setParameterStiffness(stiffness);
    // setParameterDamping(damping);    
    
    int i = 0;
    ros::Rate rate(loop_rate);
    double z_coord = panda->initialPosition.z - 0.02;
    geometry_msgs::PoseStamped externalDownMovementPositionMessage = downMovementPoseMessage(z_coord);
    
    while (ros::ok() && i < 35)
    {
        equilibriumPosePublisher.publish(externalDownMovementPositionMessage);
        rate.sleep();
        i++;
    }

    return true;
}

bool Controller::spiralMotionState()
{
    ROS_DEBUG_ONCE("In spiral motion state from controller");

    ros::Rate rate(loop_rate);
    
    Trajectory spiralTrajectory = generateArchimedeanSpiral((panda->holeDiameter / 2) * 0.001, 0.0002, 150);

    // Write to file
    writeTrajectoryToFile(spiralTrajectory, "spiral.csv");

    geometry_msgs::PoseStamped spiralMotionMessage = emptyPoseMessage();

    for (auto point : spiralTrajectory)
    {
        spiralMotionMessage = spiralPointPoseMessage(point);
        equilibriumPosePublisher.publish(spiralMotionMessage);
        panda->updatePosition(spiralMotionMessage.pose.position.x, spiralMotionMessage.pose.position.y, spiralMotionMessage.pose.position.z);
        rate.sleep();
    }
    return true;
}

bool Controller::insertionWiggleState()
{
    ROS_DEBUG_ONCE("In insertion wiggle state from controller");

    int i = 0;
    ros::Rate rate(loop_rate);

    
    double x_angle = 0.0005;
    while (ros::ok() && i < 60)
    {
        
        geometry_msgs::PoseStamped insertionWiggleMessage = insertionWigglePoseMessage(x_angle);

        equilibriumPosePublisher.publish(insertionWiggleMessage);
        rate.sleep();
        ROS_DEBUG_STREAM("Panda ring: (xyz) " << panda->orientation.x << ", 
                         "<< panda->orientation.y << ", "<< panda->orientation.z<< ", "<< panda->orientation.w);
        i++;
        if (!(i%3))
        {
            ROS_DEBUG_STREAM("x_angle flipped: x_angle = " << x_angle);
            x_angle = -(x_angle);
        }
    }

    return true;
}

bool Controller::straighteningState()
{
    ROS_DEBUG_ONCE("In straightening state from controller");

    int i = 0;
    ros::Rate rate(loop_rate);
    while (ros::ok() && i < 15)
    {
        geometry_msgs::PoseStamped straighteningMessage = straighteningPoseMessage();
        equilibriumPosePublisher.publish(straighteningMessage);
        rate.sleep();
        i++;
    }
    return true;
}

bool Controller::internalDownMovementState()
{
    ROS_DEBUG_ONCE("In internal down movement state from controller");

    Stiffness stiffness;
    Damping damping;
    stiffness.translational_x = 500;
    stiffness.translational_y = 500;
    stiffness.translational_z = 500;
    stiffness.rotational_x = 500;
    stiffness.rotational_y = 500;
    stiffness.rotational_z = 500;

    damping.translational_x = 65;
    damping.translational_y = 65;
    damping.translational_z = 65;
    damping.rotational_x = 45;
    damping.rotational_y = 45;
    damping.rotational_z = 45;

    // setParameterStiffness(stiffness);
    // setParameterDamping(damping);    
    
    int i = 0;
    ros::Rate rate(loop_rate);
    double z_coord = panda->initialPosition.z - 0.03;
    geometry_msgs::PoseStamped externalDownMovementPositionMessage = downMovementPoseMessage(z_coord);
    
    while (ros::ok() && i < 35)
    {
        equilibriumPosePublisher.publish(externalDownMovementPositionMessage);
        rate.sleep();
        i++;
    }
    return true;
}

// Private methods
void Controller::initJointTrajectoryPublisher()
{
    string topic;
    const int queueSize = 1000;

    nodeHandler->getParam("insertion/jointTrajectoryTopic", topic);
    ROS_DEBUG_STREAM("initJointTrajectoryPublisher topic: " << topic);

    jointTrajectoryPublisher = nodeHandler->advertise<trajectory_msgs::JointTrajectory>(topic, queueSize);
}

void Controller::initEquilibriumPosePublisher()
{

    string topic;
    const int queueSize = 1000;

    nodeHandler->getParam("insertion/equilibriumPoseTopic", topic);

    ROS_DEBUG_STREAM("initEquilibriumPosePublisher topic: " << topic);

    equilibriumPosePublisher = nodeHandler->advertise<geometry_msgs::PoseStamped>(topic, queueSize);
}

bool Controller::loadController(string controller)
{
    ROS_DEBUG_STREAM("In loadController()");

    string serviceName;
    nodeHandler->getParam("insertion/loadControllerService", serviceName);

    ROS_DEBUG_STREAM("Trying to load controller " << controller << " with service " << serviceName);

    ros::ServiceClient client = nodeHandler->serviceClient<controller_manager_msgs::LoadController>(serviceName);

    controller_manager_msgs::LoadController service;
    service.request.name = controller.c_str();

    if (client.call(service))
    {
        return true;
    }

    ROS_ERROR("Could not load controller %s", controller.c_str());

    return false;
}

bool Controller::switchController(string from, string to)
{
    ROS_DEBUG_STREAM("In switchController()");

    string serviceName;
    nodeHandler->getParam("insertion/switchControllerService", serviceName);

    ros::ServiceClient client = nodeHandler->serviceClient<controller_manager_msgs::SwitchController>(serviceName);

    controller_manager_msgs::SwitchController service;
    service.request.stop_controllers = {from.c_str()};
    service.request.start_controllers = {to.c_str()};
    service.request.strictness = 2;

    ROS_DEBUG_STREAM("Trying to switch controller from " << from << " to " << to << " with service " << serviceName);

    if (client.call(service))
    {
        ROS_DEBUG("Switched controller from %s to %s", from.c_str(), to.c_str());
        return true;
    }
    ROS_ERROR("Could not switch controller %s to %s", from.c_str(), to.c_str());
    return false;

}

geometry_msgs::PoseStamped Controller::initialPoseMessage()
{
    geometry_msgs::PoseStamped message = emptyPoseMessage();

    message.pose.position = panda->initialPosition;
    message.pose.orientation = panda->initialOrientation;

    return message;
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

geometry_msgs::PoseStamped Controller::downMovementPoseMessage(double z_coord)
{
    geometry_msgs::PoseStamped message = emptyPoseMessage();
    message.pose.position = panda->initialPosition;

    panda->position.z = z_coord;

    ROS_DEBUG_STREAM("Panda position z =" << panda->position.z);

    message.pose.position = panda->position;
    message.pose.orientation = panda->orientation;
    
    return message;
}

geometry_msgs::PoseStamped Controller::emptyPoseMessage()
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

geometry_msgs::PoseStamped Controller::spiralPointPoseMessage(Point point)
{
    geometry_msgs::PoseStamped message = emptyPoseMessage();

    message.pose.position.x = point.x;
    message.pose.position.y = point.y;
    message.pose.position.z = point.z;

    message.pose.orientation = panda->orientation;

    return message;
}

geometry_msgs::PoseStamped Controller::insertionWigglePoseMessage(double x_angle)
{
    geometry_msgs::PoseStamped message = emptyPoseMessage();
    message.pose.orientation = panda->orientation;
    message.pose.position = panda->position;

    // Convert to matrix
    Eigen::Affine3d tMatrix;
    tf::poseMsgToEigen(message.pose, tMatrix);

    Eigen::Affine3d rotated_tMatrix = rotateMatrixRPY(tMatrix, x_angle * M_PI, 0.0 * M_PI, 0.0 * M_PI);
    ROS_DEBUG_STREAM("rotated_tMatrix: " << endl << rotated_tMatrix.rotation());

    // Convert back to msg
    tf::poseEigenToMsg(rotated_tMatrix, message.pose);

    // Set new orientation
    panda->orientation = message.pose.orientation;

    return message;
}

geometry_msgs::PoseStamped Controller::straighteningPoseMessage()
{
    geometry_msgs::PoseStamped message = emptyPoseMessage();

    message.pose.position = panda->position;
    message.pose.orientation = panda->orientation;
    message.pose.orientation.x = 1;
    message.pose.orientation.y = 0;
    message.pose.orientation.z = 0;
    message.pose.orientation.w = 0;
    panda->orientation = message.pose.orientation;

    return message;
}

Trajectory Controller::generateArchimedeanSpiral(double a, double b, int nrOfPoints)
{
    Trajectory spiral;

    double initX = double(panda->position.x);
    double initY = double(panda->position.y);
    double initZ = double(panda->position.z);

    const double RANGE = (12 * M_PI);
    double x = initX, y = initY, z = initZ;

    for (auto i = 0; i <= nrOfPoints; i++)
    {
        Point point;

        double theta = i * (RANGE / nrOfPoints);
        double r = (a - b * theta);

        x = initX + r * cos(theta);
        y = initY + r * sin(theta);

        point.x = x;
        point.y = y;
        point.z = z;

        spiral.push_back(point);
    }
    return spiral;
}

void Controller::setParameterStiffness(Stiffness stiffness)
{
    std::vector<int> translational_stiffness, rotational_stiffness;

    translational_stiffness.push_back(stiffness.translational_x);
    translational_stiffness.push_back(stiffness.translational_y);
    translational_stiffness.push_back(stiffness.translational_z);

    rotational_stiffness.push_back(stiffness.rotational_x);
    rotational_stiffness.push_back(stiffness.rotational_y);
    rotational_stiffness.push_back(stiffness.rotational_z);

    const std::string param_translation_stiffness = "/impedance_controller/cartesian_stiffness/translation";
    const std::string param_rotation_stiffness = "/impedance_controller/cartesian_stiffness/rotation";

    nodeHandler->setParam(param_translation_stiffness, translational_stiffness); 
    nodeHandler->setParam(param_rotation_stiffness, rotational_stiffness);
}

void Controller::setParameterDamping(Damping damping)
{
    std::vector<int> translational_damping, rotational_damping;
    
    translational_damping.push_back(damping.translational_x);
    translational_damping.push_back(damping.translational_y);
    translational_damping.push_back(damping.translational_z);

    rotational_damping.push_back(damping.rotational_x);
    rotational_damping.push_back(damping.rotational_y);
    rotational_damping.push_back(damping.rotational_z);

    const std::string param_translation_damping = "/impedance_controller/cartesian_damping/translation";
    const std::string param_rotation_damping = "/impedance_controller/cartesian_damping/rotation";

    nodeHandler->setParam(param_translation_damping, translational_damping); 
}

void Controller::writeTrajectoryToFile(Trajectory trajectory, const string& fileName, bool appendToFile)
{
    ofstream outfile;

    std::stringstream filePath;
    filePath << ros::package::getPath("panda_insertion") << "/trajectories/" << fileName;

    if (appendToFile)
        outfile.open(filePath.str(), ios_base::app);
    else
        outfile.open(filePath.str());

    if (!outfile.is_open())
    {
        ROS_WARN_STREAM("Unable to open file " << filePath.str());
        return;
    }

    for (auto point : trajectory)
    {
        outfile << point.x << "," << point.y << "," << point.z << "\n";
    }

    ROS_DEBUG_STREAM("Wrote trajectory to file " << filePath.str());

    outfile.close();
}

Eigen::Affine3d Controller::rotateMatrixRPY(Eigen::Affine3d tMatrix, double rollAngle, double pitchAngle, double yawAngle)
{
    Eigen::AngleAxisd roll(rollAngle, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch(pitchAngle, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw(yawAngle, Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond quaternion = yaw * pitch * roll;


    Eigen::Affine3d rotated_tMatrix = tMatrix.rotate(quaternion);

    return rotated_tMatrix;
}
