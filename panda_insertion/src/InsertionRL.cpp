#include "panda_insertion/InsertionRL.hpp"
#include "geometry_msgs/PoseStamped.h"
#include "std_msgs/String.h"
#include <iostream>
#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <string>
#include <boost/thread.hpp>
#include <ros/duration.h>
#include <boost/algorithm/string.hpp>

using namespace std;

// Constructors
InsertionRL::InsertionRL() {}

InsertionRL::InsertionRL(ros::NodeHandle nodeHandler)
{
    ROS_DEBUG("In InsertionRL constructor");
    
    this->nodeHandler = nodeHandler;
    init();
}

// Destructor
InsertionRL::~InsertionRL() {delete tfListener;}

void InsertionRL::init()
{  
    ROS_DEBUG("In init()");

    this->tfListener = new tf2_ros::TransformListener(tfBuffer);

    transformTimer = nodeHandler.createTimer(ros::Duration(0.1), &InsertionRL::transformTimerCallback, this);
    jointStateSubscriber = nodeHandler.subscribe("/joint_states", 100, &InsertionRL::jointStatesCallback, this);
    externalForceSubscriber = nodeHandler.subscribe("/panda/franka_state_controller/F_ext", 100, &InsertionRL::externalForceCallback, this);

    observationPublisher = nodeHandler.advertise<panda_insertion::StateMsg>("insertion_spirl/state", 100);

    panda.init(&nodeHandler);
}

void InsertionRL::transformTimerCallback(const ros::TimerEvent& event)
{
    geometry_msgs::TransformStamped transformStamped;

    try
    {
        transformStamped = tfBuffer.lookupTransform("panda_link0", "tool", ros::Time::now(), ros::Duration(1.0));
    }
    catch (tf2::TransformException &ex)
    {
        ROS_WARN("%s", ex.what());
    }

    mutex.lock();
    panda.updateTransform(transformStamped.transform);
    mutex.unlock();

    this->updateRLObservation();

    //ROS_DEBUG_STREAM("transformStamped.transform: " << transformStamped.transform);
}

void InsertionRL::jointStatesCallback(const sensor_msgs::JointState& joint_state)
{
    //std::cout << "jointStatesCallback" << std::endl;
    panda.updateJointStates(joint_state);
}

void InsertionRL::externalForceCallback(const geometry_msgs::WrenchStamped::ConstPtr& message)
{
    //ROS_DEBUG_ONCE("externalForceSubscriberCallback triggered!");
    const unsigned int FREQUENCY_HZ = 100;

    //ROS_DEBUG_STREAM("force.x: " << message->wrench.force.x);

    mutex.lock();
    panda.updateWrenchForces(message->wrench);
    mutex.unlock();

    ros::Rate loop_rate(FREQUENCY_HZ);
    loop_rate.sleep();
}

void InsertionRL::updateRLObservation()
{
    // update RL observation
    //std::cout << "I am in updateRLObservation" << std::endl;
    panda_insertion::StateMsg obs_msg;
    obs_msg.q = panda.q;
    obs_msg.dq = panda.dq;
    obs_msg.ee_translation = {panda.transformStamped.transform.translation.x, 
                            panda.transformStamped.transform.translation.y,
                            panda.transformStamped.transform.translation.z};
    obs_msg.ee_rotation = {panda.transformStamped.transform.rotation.x, 
                            panda.transformStamped.transform.rotation.y,
                            panda.transformStamped.transform.rotation.z,
                            panda.transformStamped.transform.rotation.w};
    obs_msg.f_ext = panda.getWrench().force.z;
/*
    std::cout << panda.q[0] << " "
            << std::endl;
    std::cout << panda.dq[0] << " "
            << std::endl;
    std::cout << panda.transformStamped.transform.translation.x << " "
            << std::endl;
            */
    observationPublisher.publish(obs_msg);
}


