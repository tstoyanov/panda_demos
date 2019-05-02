#ifndef INSERTION_H
#define INSERTION_H
#include "panda_insertion/Controller.hpp"
#include "panda_insertion/Panda.hpp"
#include "ros/ros.h"
#include "tf2_msgs/TFMessage.h"
#include <std_srvs/Empty.h>

enum State {
    Start,
    MoveToInitialPosition,
    InitialPosition,
    ExternalDownMovement,
    SpiralMotion,
    InternalDownMovement,
    Straightening,
    InsertionWiggle
};

class Insertion
{
private:
    State activeState;
    Controller controller;
    Panda panda;

    ros::NodeHandle nodeHandler;

    // Timers
    ros::Timer periodicTimer;

    // Subscribers
    ros::Subscriber tfSubscriber;

    // Servers
    ros::ServiceServer stateMachineServer;

public:
    // Constructors
    Insertion();
    Insertion(ros::NodeHandle nodeHandler);

    // Accessors


    // Manipulators
    

    // Public methods
    void periodicTimerCallback(const ros::TimerEvent& event);
    void tfSubscriberCallback(const tf2_msgs::TFMessageConstPtr& message);

    void start();
    void moveToInitialPosition();
    void initialPosition();
    void externalDownMovement();

    void stateMachineRun();

private:
    void init();

};

#endif