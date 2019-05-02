#ifndef PANDA_H
#define PANDA_H
// #include "geometry_msgs/Point"
#include "geometry_msgs/PoseStamped.h"

class Panda
{
// Member variables
private:

public:
    geometry_msgs::Point initialPosition;
    geometry_msgs::Quaternion initialOrientation;

// Methods
public:
    // Constructors
    Panda();

    // Accessors

    // Manipulators

    // Methods
    void init();

private:
    // Methods

};

#endif