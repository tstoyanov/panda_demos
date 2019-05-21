#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "ros/ros.h"
#include "ros/duration.h"

#include "panda_insertion/Panda.hpp"
#include "panda_insertion/MessageHandler.hpp"
#include "panda_insertion/TrajectoryHandler.hpp"

#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/TwistStamped.h"
#include "panda_insertion/SwapController.h"
#include "trajectory_msgs/JointTrajectory.h"

#include <boost/thread/mutex.hpp>

#include "string"

typedef struct Stiffness
{
    int translational_x;
    int translational_y;
    int translational_z;
    int rotational_x;
    int rotational_y;
    int rotational_z;
} Stiffness;

typedef struct Damping
{
    int translational_x;
    int translational_y;
    int translational_z;
    int rotational_x;
    int rotational_y;
    int rotational_z;
} Damping;

typedef struct OrientationRPY
{
    double r, p, y;
} OrientationRPY;

class Controller
{
private:
    ros::NodeHandle* nodeHandler;
    Panda* panda;
    MessageHandler* messageHandler;
    TrajectoryHandler* trajectoryHandler;
    boost::mutex mutex;

    double loop_rate;

    ros::Publisher jointTrajectoryPublisher;
    ros::Publisher equilibriumPosePublisher;
    ros::Publisher desiredStiffnessPublisher;

    ros::ServiceServer swapControllerServer;
    ros::ServiceClient swapControllerClient;

public:
    Controller();
    ~Controller();

    void init(ros::NodeHandle*, Panda* panda);

    void startState();
    bool moveToInitialPositionState();
    bool externalDownMovementState();
    bool spiralMotionState();
    bool insertionWiggleState();
    bool straighteningState();
    bool internalDownMovementState();
    bool touchFloor();

private:
    void initJointTrajectoryPublisher();
    void initEquilibriumPosePublisher();
    void initDesiredStiffnessPublisher();

    bool loadController(std::string controller);
    bool swapControllerCallback(panda_insertion::SwapController::Request& request,
                                panda_insertion::SwapController::Response& response);

    void setParameterStiffness(Stiffness stiffness);
    void setParameterDamping(Damping damping);

    void sleepAndTell(double sleepTime);
    void setStiffness(geometry_msgs::Twist twist);
};

#endif
