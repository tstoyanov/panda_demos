#include "panda_insertion/Panda.hpp"
#include "ros/console.h"

// Constructors
Panda::Panda() : holeDiameter(16.3), endEffectorDiameter(16.25)
{
    ROS_DEBUG("Panda object created");
}

// Public methods
void Panda::init(ros::NodeHandle* nodeHandler)
{
    this->nodeHandler = nodeHandler;
    double xGoal, yGoal, zGoal;
    nodeHandler->getParam("insertion/x_goal", xGoal);
    nodeHandler->getParam("insertion/y_goal", yGoal);
    nodeHandler->getParam("insertion/z_goal", zGoal);

    ROS_DEBUG_STREAM("HEJ HEJ, xyz coming up: "<< xGoal <<", "<< yGoal << ", " << zGoal);


    initialPosition.x = xGoal;
    initialPosition.y = yGoal;
    initialPosition.z = zGoal;

    initialOrientation.x = 1.0;
    initialOrientation.y = 0.0;
    initialOrientation.z = 0.0;
    initialOrientation.w = 0.0;

    position = initialPosition;
    orientation = initialOrientation;
}

void Panda::updatePosition(double x, double y, double z)
{
    position.x = x;
    position.y = y;
    position.z = z;
}

// Private methods
