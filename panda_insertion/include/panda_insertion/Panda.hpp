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
    geometry_msgs::Point position;
    geometry_msgs::Quaternion orientation;
    const double holeDiameter;
    const double endEffectorDiameter;

// Methods
public:
    // Constructors
    Panda();

    // Accessors

    // Manipulators

    // Methods
    void init();
    void updatePosition(double x, double y, double z);

private:
    // Methods

};

#endif