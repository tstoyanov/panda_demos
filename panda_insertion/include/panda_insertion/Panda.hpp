#ifndef PANDA_H
#define PANDA_H
// #include "geometry_msgs/Point"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/WrenchStamped.h"
#include <geometry_msgs/TransformStamped.h>
#include <boost/thread/mutex.hpp>
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
    geometry_msgs::WrenchStamped wrenchMsg;
    geometry_msgs::TransformStamped transformStamped;
    const double holeDiameter;
    const double endEffectorDiameter;

    ros::NodeHandle* nodeHandler;

    boost::mutex mutex;

// Methods
public:
    // Constructor
    Panda();

    // Methods
    void init(ros::NodeHandle* nodeHandler);
    void updatePosition(double x, double y, double z);
    void updateWrenchForces(geometry_msgs::Wrench wrench);
    void updateTransform(geometry_msgs::Transform transform);
    geometry_msgs::Wrench getWrench();
};

#endif
