#include "panda_insertion/Controller.hpp"
#include "panda_insertion/Panda.hpp"

#include <iostream>
#include <fstream>
#include <stdexcept>
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
    sleepAndTell(4.0);
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

    vector<double> translationalStiffness;
    const string translationalParameter = "/move_to_initial_position/stiffness/translational";
    if (!nodeHandler->getParam(translationalParameter , translationalStiffness))
    {
        ROS_ERROR_STREAM("Could not get param: " << translationalParameter  << " from server");
        return false;
    }

    vector<double> rotationalStiffness;
    const string rotationalParameter = "/move_to_initial_position/stiffness/rotational";
    if (!nodeHandler->getParam(rotationalParameter , rotationalStiffness))
    {
        ROS_ERROR_STREAM("Could not get param: " << rotationalParameter  << " from server");
        return false;
    }

    swapControllerClient.call(swapController);
    sleepAndTell(0.5);

    // Set stiffness
    geometry_msgs::Twist twist;
    twist.linear.x = translationalStiffness.at(0);
    twist.linear.y = translationalStiffness.at(1);
    twist.linear.z = translationalStiffness.at(2);
    twist.angular.x = rotationalStiffness.at(0);
    twist.angular.y = rotationalStiffness.at(1);
    twist.angular.z = rotationalStiffness.at(2);
    setStiffness(twist);

    Trajectory initialTrajectory;
    try 
    {
        initialTrajectory = trajectoryHandler->generateInitialPositionTrajectory(100);
        //trajectoryHandler->writeTrajectoryToFile(initialTrajectory, "initTraj.csv");
    }
    catch (runtime_error e)
    {
        ROS_ERROR_STREAM(e.what());
    }

    PoseStamped initialPositionMessage = messageHandler->emptyPoseMessage();

    // Execute trajectory
    ros::Rate rate(20);
    for (auto point : initialTrajectory)
    {
        initialPositionMessage = messageHandler->pointPoseMessage(point);
        equilibriumPosePublisher.publish(initialPositionMessage);
        rate.sleep();
    }
    sleepAndTell(3.0);
    return true;
}

bool Controller::externalDownMovementState()
{
    ROS_DEBUG_ONCE("In external down movement state from controller");
    // Get parameters from server
    vector<double> translationalStiffness;
    const string translationalParameter = "/move_to_initial_position/stiffness/translational";
    if (!nodeHandler->getParam(translationalParameter , translationalStiffness))
    {
        ROS_ERROR_STREAM("Could not get param: " << translationalParameter  << " from server");
        return false;
    }

    vector<double> rotationalStiffness;
    const string rotationalParameter = "/move_to_initial_position/stiffness/rotational";
    if (!nodeHandler->getParam(rotationalParameter , rotationalStiffness))
    {
        ROS_ERROR_STREAM("Could not get param: " << rotationalParameter  << " from server");
        return false;
    }

    // Generate trajectory
    Trajectory downTrajectory;
    try
    {
        downTrajectory = trajectoryHandler->generateExternalDownTrajectory(75);
        //trajectoryHandler->writeTrajectoryToFile(downTrajectory, "downTrajectory.csv");
    }
    catch (runtime_error e)
    {
        ROS_ERROR_STREAM(e.what());
    }

    // Set stiffness
    geometry_msgs::Twist twist;
    twist.linear.x = translationalStiffness.at(0);
    twist.linear.y = translationalStiffness.at(1);
    twist.linear.z = translationalStiffness.at(2);
    twist.angular.x = rotationalStiffness.at(0);
    twist.angular.y = rotationalStiffness.at(1);
    twist.angular.z = rotationalStiffness.at(2);
    setStiffness(twist);

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
    sleepAndTell(1.0);
    return true;
}

bool Controller::spiralMotionState()
{
    ROS_DEBUG_ONCE("In spiral motion state from controller");

    // Get parameters from server
    vector<double> translationalStiffness;
    const string translationalParameter = "/spiral/stiffness/translational";
    if (!nodeHandler->getParam(translationalParameter , translationalStiffness))
    {
        ROS_ERROR_STREAM("Could not get param: " << translationalParameter  << " from server");
        return false;
    }

    vector<double> rotationalStiffness;
    const string rotationalParameter = "/spiral/stiffness/rotational";
    if (!nodeHandler->getParam(rotationalParameter, rotationalStiffness))
    {
        ROS_ERROR_STREAM("Could not get param: " << rotationalParameter  << " from server");
        return false;
    }

    // Failsafe
    geometry_msgs::Transform transform;
    mutex.lock();
    transform = panda->transformStamped.transform;
    mutex.unlock();

    const double MAX_HEIGHT = 0.05;
    if (transform.translation.z > MAX_HEIGHT)
    {
        ROS_ERROR("Tool too high, aborting spiral state.");

        return false;
    }

    // Set stiffness
    geometry_msgs::Twist twist;
    twist.linear.x = translationalStiffness.at(0);
    twist.linear.y = translationalStiffness.at(1);
    twist.linear.z = translationalStiffness.at(2);
    twist.angular.x = rotationalStiffness.at(0);
    twist.angular.y = rotationalStiffness.at(1);
    twist.angular.z = rotationalStiffness.at(2);
    setStiffness(twist);

    // Generate trajectory
    double a = (panda->holeDiameter / 2) * 0.001;
    double b = 0.0002;
    int nrOfPoints = 75;
    Trajectory spiralTrajectory = trajectoryHandler->generateArchimedeanSpiral(a, b, nrOfPoints);
    //trajectoryHandler->writeTrajectoryToFile(spiralTrajectory, "spiralMotion.csv");

    PoseStamped spiralMotionMessage = messageHandler->emptyPoseMessage();

    // Execute trajectory
    ros::Rate rate(loop_rate);
    for (auto point : spiralTrajectory)
    {
        if (inHole()) 
        {
            return true;
        }

        spiralMotionMessage = messageHandler->pointPoseMessage(point);
        equilibriumPosePublisher.publish(spiralMotionMessage);
        rate.sleep();
    }
    sleepAndTell(1.0);
    return false;
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

    // Get parameters from server
    vector<double> translationalStiffness;
    const string translationalParameter = "/wiggle/stiffness/translational";
    if (!nodeHandler->getParam(translationalParameter , translationalStiffness))
    {
        ROS_ERROR_STREAM("Could not get param: " << translationalParameter  << " from server");
        return false;
    }

    vector<double> rotationalStiffness;
    const string rotationalParameter = "/wiggle/stiffness/rotational";
    if (!nodeHandler->getParam(rotationalParameter , rotationalStiffness))
    {
        ROS_ERROR_STREAM("Could not get param: " << rotationalParameter  << " from server");
        return false;
    }

    // Set stiffness
    geometry_msgs::Twist twist;
    twist.linear.x = translationalStiffness.at(0);
    twist.linear.y = translationalStiffness.at(1);
    twist.linear.z = translationalStiffness.at(2);
    twist.angular.x = rotationalStiffness.at(0);
    twist.angular.y = rotationalStiffness.at(1);
    twist.angular.z = rotationalStiffness.at(2);
    setStiffness(twist);


    double xAng = (1.0/42.0);
    double yAng = (1.0/42.0);
    

    int i = 0;
    
    ros::Rate rate(loop_rate);

    while (ros::ok() && i < 256)
    {
        PoseStamped straighteningMessage = messageHandler->straighteningPoseMessage(xAng, yAng);
        equilibriumPosePublisher.publish(straighteningMessage);
        rate.sleep();
        i++;

        if ((i % 9) == 0)
        {
            
            ROS_DEBUG_STREAM("yAng flipped: yAng = " << yAng);
            yAng = -(yAng);
            ROS_DEBUG_STREAM("xAng flipped: xAng = " << xAng);
            xAng = -(xAng);
        }
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

    const double MAX_FORCE = 1.3;

    if (wrench.force.z > MAX_FORCE)
    {
        ROS_DEBUG("Touch floor!");
        return true;
    }

    return false;
}

bool Controller::inHole()
{
    geometry_msgs::Wrench wrench;

    mutex.lock();
    wrench = panda->wrenchMsg.wrench;
    mutex.unlock();

    const double MAX_FORCE = 1.1;

    if (wrench.force.z < MAX_FORCE || abs(wrench.force.x) > 4.0 || abs(wrench.force.y) > 4.0)
    {
        ROS_DEBUG("In hole");

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
