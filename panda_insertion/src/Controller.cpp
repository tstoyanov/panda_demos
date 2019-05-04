#include "panda_insertion/Controller.hpp"
#include "geometry_msgs/PoseStamped.h"
#include "trajectory_msgs/JointTrajectory.h"
#include "controller_manager_msgs/LoadController.h"
#include "controller_manager_msgs/SwitchController.h"
#include "panda_insertion/Panda.hpp"
#include "ros/console.h"
#include "ros/duration.h"
#include "ros/package.h"
#include "ros/time.h"
#include "stdint.h"
#include "string"
#include "vector"
#include <iostream>
#include <fstream>

using namespace std;

// Constructors
Controller::Controller() {}

// Accessors


// Manipulators


// Public methods
void Controller::init(ros::NodeHandle* nodeHandler, Panda* panda)
{
    loop_rate = 10;
    this->nodeHandler = nodeHandler;
    this->panda = panda;

    initEquilibriumPosePublisher();
    initJointTrajectoryPublisher();
}

void Controller::startState()
{
    ros::Duration(2.0).sleep();
    trajectory_msgs::JointTrajectory initialPoseMessage = initialJointTrajectoryMessage();
    jointTrajectoryPublisher.publish(initialPoseMessage);

    ros::Duration(5.0).sleep();
}

bool Controller::moveToInitialPositionState()
{
    std::vector<int> translational_stiffness, rotational_stiffness,
                     translational_damping, rotational_damping;
    translational_stiffness = {5000, 5000, 5000};
    const std::string param_translation_stiffness = "/impedance_controller/cartesian_stiffness/translation";
    const std::string param_rotation_stiffness = "/impedance_controller/cartesian_stiffness/rotation";
    const std::string param_translation_damping = "/impedance_controller/cartesian_damping/translation";
    const std::string param_rotation_damping = "/impedance_controller/cartesian_damping/rotation";
    nodeHandler->setParam(param_translation_stiffness, translational_stiffness); 
    
    if (!nodeHandler->getParam(param_translation_stiffness, translational_stiffness))
    {
        ROS_ERROR_STREAM("Parameter " << param_translation_stiffness << " not retreived");
    }
    ROS_INFO_STREAM("Translation stiffness xyz: " << translational_stiffness.at(0) << translational_stiffness.at(1) << translational_stiffness.at(2) );


    if (!loadController("impedance_controller"))
    {
        return false;
    }

    switchController("impedance_controller", "position_joint_trajectory_controller");
    ros::Duration(3.0).sleep();

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

bool Controller::initialPositionState()
{
    ROS_DEBUG_ONCE("Initial position from controller");
    ros::Duration(3).sleep();
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

    setParameterStiffness(stiffness);
    setParameterDamping(damping);    
    
    int i = 0;
    ros::Rate rate(loop_rate);
    geometry_msgs::PoseStamped externalDownMovementPositionMessage = externalDownMovementPoseMessage();
    
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
    initSpiralVector();
    
    writeSpiralToFile();

    geometry_msgs::PoseStamped spiralMotionMessage = emptyPoseMessage();

    for (auto point : spiralTrajectory)
    {
        spiralMotionMessage = spiralPointPoseMessage(point);
        equilibriumPosePublisher.publish(spiralMotionMessage);
        rate.sleep();
    }
}

// Private methods
void Controller::initJointTrajectoryPublisher()
{
    const string topic = "position_joint_trajectory_controller/command";
    const int queueSize = 1000;

    jointTrajectoryPublisher = nodeHandler->advertise<trajectory_msgs::JointTrajectory>(topic, queueSize);
}

void Controller::initEquilibriumPosePublisher()
{
    const string topic = "impedance_controller/equilibrium_pose";
    const int queueSize = 1000;

    equilibriumPosePublisher = nodeHandler->advertise<geometry_msgs::PoseStamped>(topic, queueSize);
}

bool Controller::loadController(string controller)
{
    const string serviceName = "controller_manager/load_controller";
    ros::ServiceClient client = nodeHandler->serviceClient<controller_manager_msgs::LoadController>(serviceName);

    controller_manager_msgs::LoadController service;
    service.request.name = controller.c_str();

    if (client.call(service))
    {
        ROS_DEBUG("Loaded controller %s", controller.c_str());
        return true;
    }

    ROS_ERROR("Could not load controller %s", controller.c_str());
    return false;
}

bool Controller::switchController(string from, string to)
{
    const string serviceName = "controller_manager/switch_controller";
    ros::ServiceClient client = nodeHandler->serviceClient<controller_manager_msgs::SwitchController>(serviceName);

    controller_manager_msgs::SwitchController service;
    service.request.start_controllers = {from.c_str()};
    service.request.stop_controllers = {to.c_str()};
    service.request.strictness = 3; 

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

geometry_msgs::PoseStamped Controller::externalDownMovementPoseMessage()
{
    geometry_msgs::PoseStamped message = emptyPoseMessage();
    message.pose.position = panda->initialPosition;

    panda->position.z = 0.72;

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
    geometry_msgs::PoseStamped message;
    message.pose.position.x = point.x;
    message.pose.position.y = point.y;
    message.pose.position.z = point.z;
    return message;
}

void Controller::initSpiralVector()
{
    double initX = double(panda->initialPosition.x);
    double initY = double(panda->initialPosition.y);
    double initZ = double(panda->initialPosition.z);

    ROS_DEBUG_STREAM("Init x,y,z: " << initX << ", " << initY << ", " << initZ);
    ROS_DEBUG_STREAM("Init x,y,z, panda_init: " << panda->initialPosition.x << ", " << panda->initialPosition.y<< ", " << panda->initialPosition.z);

    double radian, x = initX, y = initY, z = initZ;
    const double radianShrinkage = 0.01;
    double angle = 0;
    int a = 2, b = 2;
    const int nrOfPoints = 200;

    for (int i = 0; i < nrOfPoints; i++)
    {
        Point point;

        radian = radianShrinkage * i;
        x = (a + b * angle) * cos(angle);
        y = (a + b * angle) * sin(angle);
        z -= 0.0001;

        point.x = x;
        point.y = y;
        point.z = z;

        spiralTrajectory.push_back(point);
    }
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

void Controller::writeSpiralToFile()
{
    std::stringstream filePath;

    filePath << ros::package::getPath("panda_insertion") << "/src/spiral.csv";
    ofstream myFile(filePath.str());

    ROS_DEBUG_STREAM("Writing file to: " << filePath.str());

    if (!myFile.is_open())
    {
        ROS_WARN("Unable to open file.");
        return;
    }

    myFile << "x,y,z\n";
    for (auto point : spiralTrajectory)
    {
        myFile << point.x << "," << point.y << "," << point.z << "\n";
    }

    myFile.close();
}
// Private methods
