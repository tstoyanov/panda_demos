#include "ros/ros.h"
#include "std_msgs/String.h"
#include "geometry_msgs/PoseStamped.h"
#include "ros/duration.h"
#include "ros/console.h"
#include "panda_insertion/InsertionRL.hpp"
#include "boost/thread.hpp"

#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
    const std::string node_name = "insertion_rl";
    ros::init(argc, argv, node_name);

    ros::NodeHandle nodeHandler("");

    InsertionRL insertion_rl(nodeHandler);

    ROS_INFO("InsertionRL node running");

    ros::AsyncSpinner spinner(5);

    ROS_INFO_STREAM("Main loop in thread:" << boost::this_thread::get_id());
    spinner.start();
    
    ros::waitForShutdown();

    return 0;
}
