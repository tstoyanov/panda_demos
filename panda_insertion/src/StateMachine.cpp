#include "panda_insertion/StateMachine.hpp"
#include "geometry_msgs/PoseStamped.h"
#include <iostream>
#include <ros/ros.h>

using namespace std;

// Constructors
StateMachine::StateMachine() 
{
    activeState = Start;

}

StateMachine::StateMachine(double loop_rate)
{
    controller.setLoopRate(loop_rate);
    activeState = Start;
}

// Accessors


// Manipulators


// Public methods
void StateMachine::run()
{
    
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

        default:
        {
            ROS_DEBUG("Default");
            break;
        }
    }
}

// Private methods
void StateMachine::start()
{
    ROS_DEBUG_ONCE("In start state");
    controller.startState();
    activeState = MoveToInitialPosition;
    ROS_DEBUG("Changed state to MoveToInitialState");
}

void StateMachine::moveToInitialPosition()
{
    ROS_DEBUG_ONCE("In Move to Initial Position state");
    controller.moveToInitialPositionState();
    activeState = InitialPosition;
}

void StateMachine::initialPosition()
{
    ROS_DEBUG_ONCE("In Initial Position state");
    controller.initialPositionState();
}