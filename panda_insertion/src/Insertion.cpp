#include "panda_insertion/Insertion.hpp"
#include "geometry_msgs/PoseStamped.h"
#include "std_msgs/String.h"
#include <iostream>
#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <ros/callback_queue.h>
#include <string>
#include <boost/thread.hpp>

using namespace std;

// Constructors
Insertion::Insertion() {}

Insertion::Insertion(ros::NodeHandle nodeHandler)
{
    ROS_DEBUG("In insertion constructor");

    const double SLEEP_TIME = 4.0;
    ROS_DEBUG("Sleeping for %lf seconds", SLEEP_TIME);
    ros::Duration(SLEEP_TIME).sleep();
    
    this->nodeHandler = nodeHandler;
    init();
}

// Accessors


// Manipulators


// Public methods
void Insertion::stateMachineRun()
{
    ROS_DEBUG_STREAM_ONCE("State machine started");

    switch(activeState)
    {
        case Start:
        {
            start();
            break;
        }

        case MoveToInitialPosition:
        {
            moveToInitialPosition();
            break;
        }

         case InitialPosition:
        {
            initialPosition();
            break;
        }

        case ExternalDownMovement:
        {
            externalDownMovement();
            break;
        }

        case SpiralMotion:
        {
            ROS_DEBUG_ONCE("Spiral motion.");
            spiralMotion();
            break;
        }

        case InsertionWiggle:
        {
            ROS_DEBUG_ONCE("Insertion Wiggle");
            break;
        }

        default:
        {
            ROS_DEBUG("Default");
            break;
        }
    }
}

void Insertion::periodicTimerCallback(const ros::TimerEvent& event)
{
    ROS_DEBUG_STREAM_NAMED("thread_id" ,"Periodic timer callback in thread:" << boost::this_thread::get_id());
    stateMachineRun();
}

void Insertion::tfSubscriberCallback(const tf2_msgs::TFMessageConstPtr& message)
{
    const unsigned int FREQUENCY_HZ = 5;

    ROS_DEBUG_STREAM_NAMED("thread_id", "tf subscriber callback in thread:" << boost::this_thread::get_id());

    ros::Rate loop_rate(FREQUENCY_HZ);
    loop_rate.sleep();
}

// Private methods
void Insertion::init()
{   
    ROS_DEBUG("In init()");
    activeState = Start;

    controller.init(&nodeHandler, &panda);

    periodicTimer = nodeHandler.createTimer(ros::Duration(0.1), &Insertion::periodicTimerCallback, this);
    tfSubscriber = nodeHandler.subscribe("/tf", 1, &Insertion::tfSubscriberCallback,this);
}

void Insertion::start()
{
    ROS_DEBUG_ONCE("In start state");
    controller.startState();
    activeState = MoveToInitialPosition;
    ROS_DEBUG("Changed state to MoveToInitialState");
}

void Insertion::moveToInitialPosition()
{
    ROS_DEBUG_ONCE("In Move to Initial Position state");
    controller.moveToInitialPositionState();
    activeState = InitialPosition;
}

void Insertion::initialPosition()
{
    ROS_DEBUG_ONCE("In Initial Position state");
    controller.initialPositionState();
    activeState = ExternalDownMovement;
}

void Insertion::externalDownMovement()
{
    ROS_DEBUG_ONCE("In external down movement state");
    controller.externalDownMovementState();
    ROS_DEBUG_ONCE("External down movement DONE.");
    activeState = SpiralMotion;
}

void Insertion::spiralMotion()
{
    ROS_DEBUG_ONCE("In spiral motion state");
    controller.spiralMotionState();
    ROS_DEBUG_ONCE("Spiral motion DONE.");
    activeState = InsertionWiggle;
}