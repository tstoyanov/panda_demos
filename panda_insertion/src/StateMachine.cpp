#include "panda_insertion/StateMachine.hpp"
#include "geometry_msgs/PoseStamped.h"
#include <iostream>
#include <ros/ros.h>

using namespace std;

StateMachine::StateMachine() 
{
    activeState = Start;

}

// Methods
void StateMachine::run()
{
    
    switch(activeState)
    {

        case Start:
        {
            start();
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

// States
void StateMachine::start()
{
    ROS_DEBUG("In start state");
    activeState = InitialPosition;
    ROS_DEBUG("Changed state to initialPosition");
}

void StateMachine::initialPosition()
{
    ROS_DEBUG("In Initial Position state");
    controller.initialPosition();
}