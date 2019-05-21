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

#include "ros/console.h"
#include "ros/duration.h"
#include "ros/package.h"
#include "ros/time.h"

using namespace std;
using namespace trajectory_msgs;
using namespace geometry_msgs;
using namespace controller_manager_msgs;

// Constructors
Controller::Controller() {}

// Destructor
Controller::~Controller()
{
    delete messageHandler;
    delete trajectoryHandler;
}

// Public methods
void Controller::init(ros::NodeHandle* nodeHandler, Panda* panda)
{
    loop_rate = 10;

    this->nodeHandler = nodeHandler;
    this->panda = panda;
    this->messageHandler = new MessageHandler(nodeHandler, panda);
    this->trajectoryHandler = new TrajectoryHandler(nodeHandler, panda);

    string baseFrameId = "link0";

    initEquilibriumPosePublisher();
    initJointTrajectoryPublisher();
    initDesiredStiffnessPublisher();

    swapControllerServer = nodeHandler->advertiseService("swap_controller", &Controller::swapControllerCallback, this);
    swapControllerClient = nodeHandler->serviceClient<panda_insertion::SwapController>("swap_controller");
}

void Controller::startState()
{
    JointTrajectory initialPoseMessageJoint = messageHandler->initialJointTrajectoryMessage();
    jointTrajectoryPublisher.publish(initialPoseMessageJoint);

    sleepAndTell(4.0);
}

bool Controller::moveToInitialPositionState()
{
    // Get controllers from parameter server
    panda_insertion::SwapController swapController;
    nodeHandler->getParam("insertion/positionJointTrajectoryController", swapController.request.from);
    nodeHandler->getParam("insertion/impedanceController", swapController.request.to);

    //swapControllerClient.call(swapController);

    // Set stiffness
    geometry_msgs::Twist twist;
    double linearStiffness = 400.0;
    double angularStiffness = 30.0;
    twist.linear.x = linearStiffness;
    twist.linear.y = linearStiffness;
    twist.linear.z = linearStiffness;
    twist.angular.x = angularStiffness;
    twist.angular.y = angularStiffness;
    twist.angular.z = angularStiffness;
    setStiffness(twist);

    Trajectory initialTrajectory = trajectoryHandler->generateInitialPositionTrajectory(100);
    PoseStamped initialPositionMessage = messageHandler->emptyPoseMessage();

    trajectoryHandler->writeTrajectoryToFile(initialTrajectory, "initTraj.csv");

    // Execute trajectory
    ros::Rate rate(loop_rate);
    for (auto point : initialTrajectory)
    {
        initialPositionMessage = messageHandler->pointPoseMessage(point);
        equilibriumPosePublisher.publish(initialPositionMessage);
        rate.sleep();
    }

    return true;
}

bool Controller::externalDownMovementState()
{
    ROS_DEBUG_ONCE("In external down movement state from controller");

    Trajectory downTrajectory = trajectoryHandler->generateExternalDownTrajectory(100);
    trajectoryHandler->writeTrajectoryToFile(downTrajectory, "downTrajectory.csv");

    PoseStamped downMovementMessage = messageHandler->emptyPoseMessage();
    
    // Execute trajectory
    ros::Rate rate(loop_rate);
    for (auto point : downTrajectory)
    {
        if (touchFloor())
            break;

        downMovementMessage = messageHandler->pointPoseMessage(point);
        equilibriumPosePublisher.publish(downMovementMessage);

        rate.sleep();
    }

    return true;
}

bool Controller::spiralMotionState()
{
    ROS_DEBUG_ONCE("In spiral motion state from controller");

    // Generate trajectory
    int a = (panda->holeDiameter / 2) * 0.001;
    int b = 0.0002;
    int nrOfPoints = 150;
    Trajectory spiralTrajectory = trajectoryHandler->generateArchimedeanSpiral(a, b, nrOfPoints);

    PoseStamped spiralMotionMessage = messageHandler->emptyPoseMessage();

    // Execute trajectory
    ros::Rate rate(loop_rate);
    for (auto point : spiralTrajectory)
    {
        spiralMotionMessage = messageHandler->spiralPointPoseMessage(point);
        equilibriumPosePublisher.publish(spiralMotionMessage);
        panda->updatePosition(spiralMotionMessage.pose.position.x, spiralMotionMessage.pose.position.y, spiralMotionMessage.pose.position.z);
        rate.sleep();
    }

    return true;
}

bool Controller::insertionWiggleState()
{
    ROS_DEBUG_ONCE("In insertion wiggle state from controller");
    
    double xAngle = 0.0005;
    int i = 0;
    ros::Rate rate(loop_rate);

    while (ros::ok() && i < 60)
    {
        PoseStamped insertionWiggleMessage = messageHandler->insertionWigglePoseMessage(xAngle);

        equilibriumPosePublisher.publish(insertionWiggleMessage);
        rate.sleep();

        ROS_DEBUG_STREAM("Panda ring: (xyz) "
                         << panda->orientation.x << ", "
                         << panda->orientation.y << ", "
                         << panda->orientation.z << ", "
                         << panda->orientation.w);

        i++;

        if ((i % 3) == 0)
        {
            ROS_DEBUG_STREAM("x_angle flipped: xAngle = " << xAngle);
            xAngle = -(xAngle);
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
        PoseStamped straighteningMessage = messageHandler->straighteningPoseMessage();
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
    PoseStamped externalDownMovementPositionMessage = messageHandler->downMovementPoseMessage(z_coord);
    
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

void Controller::initDesiredStiffnessPublisher()
{
    string topic;
    const int queueSize = 100;

    nodeHandler->getParam("insertion/desiredStiffnessTopic", topic);
    ROS_DEBUG_STREAM("desiredStiffnessTopic: " << topic);

    desiredStiffnessPublisher = nodeHandler->advertise<geometry_msgs::TwistStamped>(topic, queueSize);
}

bool Controller::loadController(string controller)
{
    ROS_DEBUG_STREAM("In loadController()");

    string serviceName;
    nodeHandler->getParam("insertion/loadControllerService", serviceName);

    ROS_DEBUG_STREAM("Trying to load controller " << controller << " with service " << serviceName);

    ros::ServiceClient client = nodeHandler->serviceClient<LoadController>(serviceName);

    LoadController service;
    service.request.name = controller.c_str();

    if (client.call(service))
    {
        return true;
    }

    ROS_ERROR("Could not load controller %s", controller.c_str());

    return false;
}

bool Controller::swapControllerCallback(panda_insertion::SwapController::Request& request,
                                        panda_insertion::SwapController::Response& response)
{
    ROS_DEBUG_STREAM("In swapController callback");

    const string from = request.from;
    const string to = request.to;

    if (!loadController(to))
    {
        return false;
    }

    string serviceName;
    nodeHandler->getParam("insertion/switchControllerService", serviceName);

    ros::ServiceClient client = nodeHandler->serviceClient<SwitchController>(serviceName);

    SwitchController service;
    service.request.stop_controllers = {from.c_str()};
    service.request.start_controllers = {to.c_str()};
    service.request.strictness = 2;

    ROS_DEBUG_STREAM("Trying to swap controller from [" << from << "] "
           << " to [" << to << "] with service [" << serviceName << "]");

    if (client.call(service))
    {
        ROS_DEBUG("Called service to swap controller from %s to %s", from.c_str(), to.c_str());
        return true;
    }

    ROS_ERROR("Could not call service to swap controller %s to %s", from.c_str(), to.c_str());

    return false;
}

void Controller::setParameterStiffness(Stiffness stiffness)
{
    vector<int> translational_stiffness, rotational_stiffness;

    translational_stiffness.push_back(stiffness.translational_x);
    translational_stiffness.push_back(stiffness.translational_y);
    translational_stiffness.push_back(stiffness.translational_z);

    rotational_stiffness.push_back(stiffness.rotational_x);
    rotational_stiffness.push_back(stiffness.rotational_y);
    rotational_stiffness.push_back(stiffness.rotational_z);

    const string param_translation_stiffness = "/impedance_controller/cartesian_stiffness/translation";
    const string param_rotation_stiffness = "/impedance_controller/cartesian_stiffness/rotation";

    nodeHandler->setParam(param_translation_stiffness, translational_stiffness); 
    nodeHandler->setParam(param_rotation_stiffness, rotational_stiffness);
}

void Controller::setParameterDamping(Damping damping)
{
    vector<int> translational_damping, rotational_damping;
    
    translational_damping.push_back(damping.translational_x);
    translational_damping.push_back(damping.translational_y);
    translational_damping.push_back(damping.translational_z);

    rotational_damping.push_back(damping.rotational_x);
    rotational_damping.push_back(damping.rotational_y);
    rotational_damping.push_back(damping.rotational_z);

    const string param_translation_damping = "/impedance_controller/cartesian_damping/translation";
    const string param_rotation_damping = "/impedance_controller/cartesian_damping/rotation";

    nodeHandler->setParam(param_translation_damping, translational_damping); 
}

void Controller::sleepAndTell(double sleepTime)
{
    ROS_DEBUG("Sleeping for %lf seconds", sleepTime);
    ros::Duration(sleepTime).sleep();
}

bool Controller::touchFloor()
{
    geometry_msgs::Wrench wrench;

    mutex.lock();
    wrench = panda->wrenchMsg.wrench;
    mutex.unlock();

    const double MAX_FORCE = 2.0;

    if (wrench.force.z > MAX_FORCE)
    {
        ROS_DEBUG("Touch floor!");
        return true;
    }

    return false;
}

void Controller::setStiffness(geometry_msgs::Twist twist)
{
    geometry_msgs::TwistStamped twistStamped;
 
    // Set header
    string frameId = "";
    ros::Time stamp(0.0);
    uint32_t seq = 0;
    twistStamped.header.frame_id = frameId;
    twistStamped.header.stamp = stamp;
    twistStamped.header.seq = seq;

    // Set stiffness
    twistStamped.twist = twist;

    desiredStiffnessPublisher.publish(twistStamped);
}
