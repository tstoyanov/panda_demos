#include "ros/ros.h"
#include "std_msgs/String.h"
#include "trajectory_msgs/JointTrajectory.h"
#include <std_msgs/Float64.h>

#include <sstream>
#include <vector>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "talker");

    ros::NodeHandle nodeHandle;
    const std::string topic = "position_joint_trajectory_controller/command";

    ros::Publisher publisher = nodeHandle.advertise<trajectory_msgs::JointTrajectory>(topic, 1000);

    ros::Rate loop_rate(10);

    while (ros::ok())
    {
        ROS_INFO("Insertion says hello.");

        trajectory_msgs::JointTrajectory message;

        publisher.publish(message);

        loop_rate.sleep();
    }

    return 0;
}
