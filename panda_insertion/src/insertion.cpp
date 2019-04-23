#include "ros/ros.h"
#include "std_msgs/String.h"
#include "geometry_msgs/PoseStamped.h"
#include "panda_insertion/StateMachine.hpp"

#include <sstream>
#include <vector>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "insertion");

    ros::NodeHandle nodeHandle;
    const std::string topic = "impedance_controller/equilibrium_pose";
    ros::Publisher publisher = nodeHandle.advertise<geometry_msgs::PoseStamped>(topic, 1000);

    ros::Rate loop_rate(10);

    StateMachine stateMachine;

    while (ros::ok())
    {
        stateMachine.run();

        loop_rate.sleep();
    }

    return 0;
}
