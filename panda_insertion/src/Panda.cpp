#include "panda_insertion/Panda.hpp"
#include "ros/console.h"

// Constructors
Panda::Panda() : holeDiameter(16.3), endEffectorDiameter(16.25)
{
    ROS_DEBUG("Panda object created");
    init();
}

// Public methods
void Panda::init()
{
    initialPosition.x = 0.475;
    initialPosition.y = 0.105;
    initialPosition.z = 0.74;

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
