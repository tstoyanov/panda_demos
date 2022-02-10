#include "panda_insertion/Panda.hpp"
#include "ros/console.h"

// Constructors
Panda::Panda() : holeDiameter(16.3), endEffectorDiameter(16.25)
{
    ROS_DEBUG("Panda object created");
    q.reserve(7);
    dq.reserve(7);
}

// Public methods
void Panda::init(ros::NodeHandle* nodeHandler)
{
    this->nodeHandler = nodeHandler;
    double xGoal, yGoal, zGoal;
    nodeHandler->getParam("insertion/x_goal", xGoal);
    nodeHandler->getParam("insertion/y_goal", yGoal);
    nodeHandler->getParam("insertion/z_goal", zGoal);

    ROS_DEBUG_STREAM("Initial goal xyz: "<< xGoal <<", "<< yGoal << ", " << zGoal);

    initialPosition.x = xGoal;
    initialPosition.y = yGoal;
    initialPosition.z = zGoal;

    initialOrientation.x = 0.983;
    initialOrientation.y = 0.186;
    initialOrientation.z = 0.002;
    initialOrientation.w = 0.001;

    position = initialPosition;
    orientation = initialOrientation;

    // Compensation of error to desired pose
    straightOrientation.x = -0.999726;
    straightOrientation.y = 0.014363;
    straightOrientation.z = -0.0171982;
    straightOrientation.w = 0.00678687;
}

void Panda::updatePosition(double x, double y, double z)
{
    position.x = x;
    position.y = y;
    position.z = z;
}

void Panda::updateWrenchForces(geometry_msgs::Wrench wrench)
{
    mutex.lock();
    this->wrenchMsg.wrench = wrench;
    mutex.unlock();
}

geometry_msgs::Wrench Panda::getWrench()
{
    return wrenchMsg.wrench;
}

void Panda::updateTransform(geometry_msgs::Transform transform)
{
    transformStamped.transform = transform;
}

void Panda::updateJointStates(sensor_msgs::JointState joint_state)
{
    q = joint_state.position;
    dq = joint_state.velocity;
}

// Private methods
