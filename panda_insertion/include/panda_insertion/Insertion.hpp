#ifndef INSERTION_H
#define INSERTION_H
#include "panda_insertion/Controller.hpp"
#include "panda_insertion/Panda.hpp"
#include "panda_insertion/ChangeState.h"

#include "ros/ros.h"

#include "tf2_msgs/TFMessage.h"
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


    ros::NodeHandle nodeHandler;

    // Timers
    ros::Timer periodicTimer;

    // Subscribers
    ros::Subscriber tfSubscriber;

    // Servers and clients
    ros::ServiceServer iterateStateServer;
    ros::ServiceClient stateClient;

public:
    // Constructors
    Insertion();
    Insertion(ros::NodeHandle nodeHandler);

    // Public methods
    void periodicTimerCallback(const ros::TimerEvent& event);
    void tfSubscriberCallback(const tf2_msgs::TFMessageConstPtr& message);
    bool changeStateCallback(panda_insertion::ChangeState::Request& request, panda_insertion::ChangeState::Response& response);
    void changeState(std::string stateName);

    void start();
    void moveToInitialPosition();
    void externalDownMovement();
    void spiralMotion();
    void straightening();
    void insertion();
    void internalDownMovement();
    void finish();
    void idle();

    void stateMachineRun();

private:
    void init();

};

#endif