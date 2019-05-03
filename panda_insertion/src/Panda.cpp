#include "panda_insertion/Panda.hpp"
#include "ros/console.h"

// Constructors
Panda::Panda()
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

// Private methods
