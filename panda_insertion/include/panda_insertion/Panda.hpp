#ifndef PANDA_H
#define PANDA_H
// #include "geometry_msgs/Point"
#include "geometry_msgs/PoseStamped.h"
#include "ros/ros.h"

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

    ros::NodeHandle* nodeHandler;

// Methods
public:
    // Constructor
    Panda();

    // Methods
    void init(ros::NodeHandle* nodeHandler);
    void updatePosition(double x, double y, double z);
};

#endif