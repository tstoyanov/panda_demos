#include "panda_insertion/Insertion.hpp"
#include "geometry_msgs/PoseStamped.h"
#include "std_msgs/String.h"
#include <iostream>
#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <ros/callback_queue.h>
#include <string>
#include <boost/thread.hpp>
#include <boost/algorithm/string.hpp>

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
            insertion();
            break;
        }
        case Straightening:
        {
            ROS_DEBUG_ONCE("Straightening");
            straightening();
            break;
        }

        case InternalDownMovement:
        {
            ROS_DEBUG_ONCE("Internal down movement");
            internalDownMovement();
            break;
        }

        case Finish:
        {
            ROS_DEBUG_ONCE("Finish state");
            finish();
            break;
        }

        case Idle:
        {
            idle();
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

bool Insertion::changeStateCallback(panda_insertion::ChangeState::Request& request, panda_insertion::ChangeState::Response& response)
{
    const string state = request.req;
    
    ROS_DEBUG_STREAM("State: " << state);
    if (boost::iequals(state, "Start"))
    {
        mutex.lock();
        activeState = Start;
        mutex.unlock();
    }
    if (boost::iequals(state, "MoveToInitialPosition"))
    {
        mutex.lock();
        activeState = MoveToInitialPosition;
        mutex.unlock();
    }
    if (boost::iequals(state, "InitialPosition"))
    {
        mutex.lock();
        activeState = InitialPosition;
        mutex.unlock();
    }
    if (boost::iequals(state, "ExternalDownMovement"))
    {
        mutex.lock();
        activeState = ExternalDownMovement;
        mutex.unlock();
    }
    if (boost::iequals(state, "SpiralMotion"))
    {
        mutex.lock();
        activeState = SpiralMotion;
        mutex.unlock();
    }
    if (boost::iequals(state, "InternalDownMovement"))
    {
        mutex.lock();
        activeState = InternalDownMovement;
        mutex.unlock();
    }
    if (boost::iequals(state, "Straightening"))
    {
        mutex.lock();
        activeState = Straightening;
        mutex.unlock();
    }
    if (boost::iequals(state, "InsertionWiggle"))
    {
        mutex.lock();
        activeState = InsertionWiggle;
        mutex.unlock();
    }
    if (boost::iequals(state, "Finish"))
    {
        mutex.lock();
        activeState = Finish;
        mutex.unlock();
    }
    if (boost::iequals(state, "Idle"))
    {
        mutex.lock();
        activeState = Idle;
        mutex.unlock();
    }

    return true;
}

void Insertion::changeState(string stateName)
{
    panda_insertion::ChangeState state;
    state.request.req = stateName;
    stateClient.call(state);
}

void Insertion::init()
{   
    ROS_DEBUG("In init()");
    activeState = Start;

    controller.init(&nodeHandler, &panda);
    panda.init(&nodeHandler);

    periodicTimer = nodeHandler.createTimer(ros::Duration(0.1), &Insertion::periodicTimerCallback, this);
    tfSubscriber = nodeHandler.subscribe("/tf", 1, &Insertion::tfSubscriberCallback,this);
    iterateStateServer = nodeHandler.advertiseService("change_state", &Insertion::changeStateCallback, this);
    stateClient = nodeHandler.serviceClient<panda_insertion::ChangeState>("change_state");
}

void Insertion::start()
{
    ROS_DEBUG_ONCE("In start state");
    controller.startState();
    changeState("idle");
}

void Insertion::moveToInitialPosition()
{
    ROS_DEBUG_ONCE("In Move to Initial Position state");
    controller.moveToInitialPositionState();
    changeState("idle");
}

void Insertion::initialPosition()
{
    ROS_DEBUG_ONCE("In Initial Position state");
    controller.initialPositionState();
    changeState("idle");
}

void Insertion::externalDownMovement()
{
    ROS_DEBUG_ONCE("In external down movement state");
    controller.externalDownMovementState();
    ROS_DEBUG_ONCE("External down movement DONE.");
    changeState("idle");
}

void Insertion::spiralMotion()
{
    ROS_DEBUG_ONCE("In spiral motion state");
    controller.spiralMotionState();
    ROS_DEBUG_ONCE("Spiral motion DONE.");
    changeState("idle");
}

void Insertion::insertion()
{
    ROS_DEBUG_ONCE("In insertion state");
    controller.insertionWiggleState();
    ROS_DEBUG_ONCE("Insertion DONE.");
    changeState("idle");
}

void Insertion::straightening()
{
    ROS_DEBUG_ONCE("In straightening state");
    controller.straighteningState();
    ROS_DEBUG_ONCE("Straightening DONE.");
    changeState("idle");
}

void Insertion::internalDownMovement()
{
    ROS_DEBUG_ONCE("In internal down movement state");
    controller.internalDownMovementState();
    ROS_DEBUG_ONCE("Internal down movement DONE.");
    changeState("idle");
}

void Insertion::finish()
{
    ROS_INFO_ONCE("Insertion completed.");
}

void Insertion::idle()
{
    ROS_DEBUG_ONCE("In idle state");
}