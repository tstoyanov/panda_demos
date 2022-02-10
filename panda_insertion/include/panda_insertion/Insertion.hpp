#ifndef INSERTION_H
#define INSERTION_H
#include "panda_insertion/Controller.hpp"
#include "panda_insertion/Panda.hpp"
#include "panda_insertion/ChangeState.h"

#include "ros/ros.h"

#include "tf2_msgs/TFMessage.h"
#include <tf2_ros/transform_listener.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

#include <geometry_msgs/TransformStamped.h>
#include "geometry_msgs/WrenchStamped.h"

#include <std_srvs/Empty.h>

#include <boost/thread/mutex.hpp>

enum State {
    Start,
    MoveToInitialPosition,
    ExternalDownMovement,
    SpiralMotion,
    InternalDownMovement,
    Straightening,
    InsertionWiggle,
    InternalUpMovement,
    Finish,
    Idle,
    END_OF_STATES
};

class Insertion
{
private:
    State activeState;
    Controller controller;
    Panda panda;
    boost::mutex mutex;
    int state_machine_loop;

    ros::NodeHandle nodeHandler;

    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener* tfListener;

    // Timers
    ros::Timer periodicTimer;
    ros::Timer transformTimer;

    // Subscribers
    ros::Subscriber tfSubscriber;
    ros::Subscriber externalForceSubscriber;
    ros::Subscriber jointStateSubscriber;

    // Servers and clients
    ros::ServiceServer iterateStateServer;
    ros::ServiceClient stateClient;

public:
    // Constructors
    Insertion();
    Insertion(ros::NodeHandle nodeHandler);

    // Destructor
    ~Insertion();

    // Public methods
    void periodicTimerCallback(const ros::TimerEvent& event);
    void transformTimerCallback(const ros::TimerEvent& event);
    void tfSubscriberCallback(const tf2_msgs::TFMessageConstPtr& message);
    void externalForceSubscriberCallback(const geometry_msgs::WrenchStampedConstPtr& message);
    bool changeStateCallback(panda_insertion::ChangeState::Request& request, panda_insertion::ChangeState::Response& response);
    void changeState(std::string stateName);

    void jointStatesCallback(const sensor_msgs::JointState& joint_states);

    void start();
    void moveToInitialPosition();
    void externalDownMovement();
    void spiralMotion();
    void straightening();
    void insertionWiggle();
    void internalDownMovement();
    void internalUpMovement();
    void finish();
    void idle();

    void stateMachineRun();

private:
    void init();

};

#endif
