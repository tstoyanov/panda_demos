#include "ros/ros.h"
#include "std_msgs/String.h"
#include "geometry_msgs/PoseStamped.h"
#include "ros/duration.h"
#include "panda_insertion/StateMachine.hpp"

#include <sstream>
#include <vector>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "insertion");

    StateMachine stateMachine;
    ros::Duration(8.0).sleep();
    ros::Rate loop_rate(10);

    while (ros::ok())
    {
        stateMachine.run();

        loop_rate.sleep();
    }

    return 0;
}
