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

    double frequency = 10;
    
    StateMachine stateMachine(frequency);
    ros::Rate loop_rate(frequency);

    ros::Duration(3.0).sleep();

    while (ros::ok())
    {
        stateMachine.run();

        loop_rate.sleep();
    }

    return 0;
}
