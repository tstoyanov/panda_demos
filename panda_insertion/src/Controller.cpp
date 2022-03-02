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
#include "controller_manager_msgs/UnloadController.h"
#include "controller_manager_msgs/SwitchController.h"

#include "ros/console.h"
#include "ros/duration.h"
#include "ros/package.h"
#include "ros/time.h"

using namespace std;
using namespace trajectory_msgs;
//using namespace geometry_msgs;
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
    reward = 0.0;

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

void Controller::swapToEffortController()
{
    // Get controllers from parameter server
    panda_insertion::SwapController swapController;
    nodeHandler->getParam("insertion/impedanceController", swapController.request.from);
    nodeHandler->getParam("insertion/positionJointTrajectoryController",swapController.request.to);

    swapControllerClient.call(swapController);
    sleepAndTell(0.5);
}

void Controller::swapToImpedanceController()
{
    // Get controllers from parameter server
    panda_insertion::SwapController swapController;
    nodeHandler->getParam("insertion/positionJointTrajectoryController", swapController.request.from);
    nodeHandler->getParam("insertion/impedanceController", swapController.request.to);

    swapControllerClient.call(swapController);
    sleepAndTell(0.5);
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
    swapToImpedanceController();
    sleepAndTell(0.5);

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
        initialTrajectory = trajectoryHandler->generateInitialPositionTrajectory(200);
        //trajectoryHandler->writeTrajectoryToFile(initialTrajectory, twist, "actions_terminals_infos.csv", true);
    }
    catch (runtime_error e)
    {
        ROS_ERROR_STREAM(e.what());
    }

    geometry_msgs::PoseStamped initialPositionMessage = messageHandler->emptyPoseMessage();
    
    // Execute trajectory
    //reward = 0;
    ros::Rate rate(20);
    int i = 0;
    bool bTerminal = false;
    for (auto point : initialTrajectory)
    {
        i++;
        //if (i == initialTrajectory.size())
        {
            //bTerminal = true;
            //reward += 1.0;
        }
        
        // write dataset of state, action, reward, terminal
        //trajectoryHandler->writeDataset("peg_in_hole_dataset.csv", point, twist, reward, bTerminal, true);

        // publish way point
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
    Trajectory downTrajectory;
    try
    {
        downTrajectory = trajectoryHandler->generateExternalDownTrajectory(75);
        //trajectoryHandler->writeTrajectoryToFile(downTrajectory, twist, "actions_terminals_infos.csv", true);
    }
    catch (runtime_error e)
    {
        ROS_ERROR_STREAM(e.what());
    }

    geometry_msgs::PoseStamped downMovementMessage = messageHandler->emptyPoseMessage();
    
    // Execute trajectory
    ros::Rate rate(loop_rate);
    int i = 0;
    reward = 0;
    bool bTerminal = false;
    for (auto point : downTrajectory)
    {
        i++;
        if (touchFloor())
        {
            //bTerminal = true;
            reward += 1.0;
            // write dataset of state, action, reward, terminal
            trajectoryHandler->writeDataset("peg_in_hole_dataset.csv", point, twist, reward, bTerminal, true);
            break;
        }

        //if (i == downTrajectory.size())
        //{
        //    bTerminal = true;
        //}
        
        // write dataset of state, action, reward, terminal
        trajectoryHandler->writeDataset("peg_in_hole_dataset.csv", point, twist, reward, bTerminal, true);
        
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

    std::cout << "+++spiral z:" << transform.translation.z << std::endl; 
    const double MAX_HEIGHT = 0.06;
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
    //double a = (panda->holeDiameter / 2) * 0.001;

    double a = 0.0;
    double b = 0.35 * 0.001;
    int nrOfPoints = 250;

    Trajectory spiralTrajectory = trajectoryHandler->generateArchimedeanSpiral(a, b, nrOfPoints);
    //trajectoryHandler->writeTrajectoryToFile(spiralTrajectory, twist, "actions_terminals_infos.csv", true);

    Trajectory spiralTrajectory1 = trajectoryHandler->generateArchimedeanSpiral(0.0, 0.34*0.0015, nrOfPoints);
    
    Trajectory spiralTrajectory2 = trajectoryHandler->generateArchimedeanSpiral(0.01, 0.345*0.002, nrOfPoints);
    
    Trajectory spiralTrajectory3 = trajectoryHandler->generateArchimedeanSpiral(0.005, 0.32*0.003, nrOfPoints);
    
    trajectoryHandler->writeSpiralTrajectories(spiralTrajectory, spiralTrajectory1, spiralTrajectory2, spiralTrajectory3, "spiral_trajectories.csv", true);
    
    
    geometry_msgs::PoseStamped spiralMotionMessage = messageHandler->emptyPoseMessage();

    // Execute trajectory
    ros::Rate rate(loop_rate);
    int i = 0;
    bool bTerminal = false;
    for (auto point : spiralTrajectory)
    {
        i++;
        
        spiralMotionMessage = messageHandler->pointPoseMessage(point);
        // add sine Yaw
        double theta_z = messageHandler->sineYaw(spiralMotionMessage, i);
        //std::cout << "theta_z in contrller:" << theta_z << std::endl;
        //std::cout << "======spiral motion message:" << spiralMotionMessage << std::endl;

        if (inHole()) 
        {
            bTerminal = true;
            reward += 1.0;
            // write dataset of state, action, reward, terminal
            //trajectoryHandler->writeDataset("peg_in_hole_dataset.csv", point, twist, reward, bTerminal, true);
            trajectoryHandler->writeSpiralDataset("peg_in_hole_dataset.csv", spiralMotionMessage, theta_z, twist, reward, bTerminal, true);
            
            return true;
        }

        // write dataset of state, action, reward, terminal
        //trajectoryHandler->writeDataset("peg_in_hole_dataset.csv", point, twist, reward, bTerminal, true);
        trajectoryHandler->writeSpiralDataset("peg_in_hole_dataset.csv", spiralMotionMessage, theta_z, twist, reward, bTerminal, true);

        equilibriumPosePublisher.publish(spiralMotionMessage);
        rate.sleep();
    }
    sleepAndTell(1.0);
    return false;
}

bool Controller::insertionWiggleState()
{
    ROS_DEBUG_ONCE("In wiggle state from controller");

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

    double xAng = (1.0 / 70.0);
    double yAng = (1.0 / 70.0);

    int i = 0;
    ros::Rate rate(loop_rate);

    while (ros::ok() && i < 150)
    {
        ROS_DEBUG_STREAM("i: " << i);
        ROS_DEBUG_STREAM("xAng: " << xAng << ", yAng: " << yAng);

        geometry_msgs::PoseStamped wiggleMessage = messageHandler->insertionWigglePoseMessage(xAng, yAng);
        equilibriumPosePublisher.publish(wiggleMessage);
        rate.sleep();
        i++;

        if ((i % 10) == 0)
        {
            xAng = xAng * 0.97;
            yAng = yAng * 0.97;

            ROS_DEBUG_STREAM("yAng flipped: yAng = " << yAng);
            yAng = -(yAng);
            ROS_DEBUG_STREAM("xAng flipped: xAng = " << xAng);
            xAng = -(xAng);
        }
    }

    return true;
}

bool Controller::straighteningState()
{
    ROS_DEBUG_ONCE("In straightening state from controller");

    // Get parameters from server
    vector<double> translationalStiffness;
    const string translationalParameter = "/straightening/stiffness/translational";
    if (!nodeHandler->getParam(translationalParameter , translationalStiffness))
    {
        ROS_ERROR_STREAM("Could not get param: " << translationalParameter  << " from server");
        return false;
    }

    vector<double> rotationalStiffness;
    const string rotationalParameter = "/straightening/stiffness/rotational";
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

    ros::Rate rate(loop_rate);
    int i = 0;
    bool bTerminal = false;
    geometry_msgs::PoseStamped poseMessage = messageHandler->straighteningPoseMessage();
    while (ros::ok() && i < 20)
    {
        //if (i == 19)
        {
            //bTerminal = true;
            //reward += 1.0;
        }
        Point point;
        point.x = poseMessage.pose.position.x;
        point.y = poseMessage.pose.position.y;
        point.z = poseMessage.pose.position.z;

        // write dataset of state, action, reward, terminal
        //trajectoryHandler->writeDataset("peg_in_hole_dataset.csv", point, twist, reward, bTerminal, true);

        equilibriumPosePublisher.publish(poseMessage);
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
    geometry_msgs::PoseStamped externalDownMovementPositionMessage = messageHandler->downMovementPoseMessage(z_coord);
    
    while (ros::ok() && i < 35)
    {
        equilibriumPosePublisher.publish(externalDownMovementPositionMessage);
        rate.sleep();
        i++;
    }

    return true;
}

bool Controller::internalUpMovementState()
{
    ROS_DEBUG_ONCE("In internal up movement state from controller");
   // Get parameters from server
        vector<double> translationalStiffness;
    const string translationalParameter = "/internal_up_movement/stiffness/translational";
    if (!nodeHandler->getParam(translationalParameter , translationalStiffness))
    {
        ROS_ERROR_STREAM("Could not get param: " << translationalParameter  << " from server");
        return false;
    }

    vector<double> rotationalStiffness;
    const string rotationalParameter = "/internal_up_movement/stiffness/rotational";
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

    // Generate trajectory
    Trajectory upTrajectory;
    try
    {
        upTrajectory = trajectoryHandler->generateInternalUpTrajectory(100);
        //trajectoryHandler->writeTrajectoryToFile(upTrajectory, twist, "actions_terminals_infos.csv", true, true);
    }
    catch (runtime_error e)
    {
        ROS_ERROR_STREAM(e.what());
    }

    geometry_msgs::PoseStamped upMovementMessage = messageHandler->emptyPoseMessage();

    // Execute trajectory
    ros::Rate rate(loop_rate);
    int i = 0;
    bool bTerminal = false;
    for (auto point : upTrajectory)
    {
        i++;
        //if (i == upTrajectory.size())
        {
            //bTerminal = true;
            //reward += 1.0;
        }
        
        // write dataset of state, action, reward, terminal
        //trajectoryHandler->writeDataset("peg_in_hole_dataset.csv", point, twist, reward, bTerminal, true);

        upMovementMessage = messageHandler->pointPoseMessage(point);
        equilibriumPosePublisher.publish(upMovementMessage);

        rate.sleep();
    }
    sleepAndTell(1.0);
    return true;
}

bool Controller::idleState()
{
    geometry_msgs::Transform transform;

    mutex.lock();
    transform = panda->transformStamped.transform;
    mutex.unlock();

    ROS_DEBUG_STREAM("\ntranslation.x: " << transform.translation.x << endl <<
                    "translation.y: " << transform.translation.y << endl <<  
                    "translation.z: " << transform.translation.z << endl);

    swapToEffortController();

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

bool Controller::unloadController(string controller)
{
    ROS_DEBUG_STREAM("In unloadController()");

    string serviceName;
    nodeHandler->getParam("insertion/unloadControllerService", serviceName);

    ROS_DEBUG_STREAM("Trying to unload controller " << controller << " with service " << serviceName);

    ros::ServiceClient client = nodeHandler->serviceClient<UnloadController>(serviceName);

    UnloadController service;
    service.request.name = controller.c_str();

    if (client.call(service))
    {
        return true;
    }

    ROS_ERROR("Could not unload controller %s", controller.c_str());

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

    const double MAX_FORCE = 1.5;

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
    const geometry_msgs::Transform transMsg = panda->transformStamped.transform;
    mutex.unlock();

    const double MAX_FORCE = 1.8;

    ROS_DEBUG_STREAM("z: " << transMsg.translation.z);

    if (transMsg.translation.z < 0.04)
    {
        ROS_DEBUG_STREAM("In hole, x-force: " << wrench.force.x << ", y-force: " << wrench.force.y
                         << "z-force: " << wrench.force.z);

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

//void Controller::matrixDifference(Eigen::Affine3d currentPose, Eigen::Affine3d desiredPose)
void Controller::matrixDifference()
{
    // Test case ---->
    geometry_msgs::Pose robotPoseMsg = messageHandler->generateRobotPoseMessage();
    geometry_msgs::Pose desiredPoseMsg = messageHandler->generateRobotErrorPoseMessage();

    Eigen::Affine3d currentPoseMatrix;
    Eigen::Affine3d desiredPoseMatrix;
    // <-----

    // Convert pose messages to transformation matrices
    tf::poseMsgToEigen(robotPoseMsg, currentPoseMatrix);
    tf::poseMsgToEigen(desiredPoseMsg, desiredPoseMatrix);

    // Extract translation
    Eigen::Vector3d currentTranslation = currentPoseMatrix.translation();
    Eigen::Vector3d desiredTranslation = desiredPoseMatrix.translation();
    
    // Extract rotation
    Eigen::Matrix3d currentRotation = currentPoseMatrix.rotation();
    Eigen::Matrix3d desiredRotation = desiredPoseMatrix.rotation();

    // Calculate difference in translation
    Eigen::Vector3d errorTranslation = desiredTranslation - currentTranslation;
    auto errorTranslationNorm = desiredTranslation.norm();

    // Calculate difference in rotation
    //Eigen::Matrix3d errorRotation = currentRotation.transpose() * desiredRotation;
    Eigen::Matrix3d errorRotation = desiredRotation * currentRotation.transpose();

    // Convert error rotation to angle-axis representation
    Eigen::AngleAxisd angleAxisRotation(errorRotation);
    auto angle = angleAxisRotation.angle();
    auto axis = angleAxisRotation.axis();

    ROS_DEBUG("\n----------------------\n");
    ROS_DEBUG_STREAM("errorTranslation: " << endl << errorTranslation);
    ROS_DEBUG_STREAM("errorTranslationNorm : " << errorTranslationNorm);
    ROS_DEBUG_STREAM("errorRotation: " << endl << errorRotation);
    ROS_DEBUG_STREAM("angle: " << endl << angle);
    ROS_DEBUG_STREAM("axis: " << endl << axis << endl);
}
