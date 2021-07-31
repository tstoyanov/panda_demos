#ifndef INSERTION_RL_H
#define INSERTION_RL_H
#include "panda_insertion/Panda.hpp"

#include "ros/ros.h"
#include <tf2_ros/transform_listener.h>

#include <geometry_msgs/TransformStamped.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/WrenchStamped.h>

#include <boost/thread/mutex.hpp>
#include "panda_insertion/StateMsg.h"

class InsertionRL
{
private:
    Panda panda;
    boost::mutex mutex;

    ros::NodeHandle nodeHandler;
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener* tfListener;
    ros::Timer transformTimer;

    // Subscribers
    ros::Subscriber jointStateSubscriber;
    ros::Subscriber externalForceSubscriber;

    // Publisher
    ros::Publisher observationPublisher;

public:
    // Constructors
    InsertionRL();
    InsertionRL(ros::NodeHandle nodeHandler);

    // Destructor
    ~InsertionRL();

    void transformTimerCallback(const ros::TimerEvent& event);
    void jointStatesCallback(const sensor_msgs::JointState& joint_states);
    void externalForceCallback(const geometry_msgs::WrenchStampedConstPtr& message);

    void updateRLObservation();

private:
    void init();

};

#endif
