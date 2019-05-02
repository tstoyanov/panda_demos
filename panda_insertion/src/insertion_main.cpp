#include "ros/ros.h"
#include "std_msgs/String.h"
#include "geometry_msgs/PoseStamped.h"
#include "ros/duration.h"
#include "ros/console.h"
#include "panda_insertion/Insertion.hpp"
#include "boost/thread.hpp"

#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
    const std::string node_name = "insertion";
    ros::init(argc, argv, node_name);

    ros::NodeHandle nodeHandler("");

    Insertion insertion(nodeHandler);

    // ros::Duration(4.0).sleep();

    ROS_INFO("Insertion node running");

    ros::AsyncSpinner spinner(2); // Use 2 threads

    ROS_INFO_STREAM("Main loop in thread:" << boost::this_thread::get_id());
    spinner.start();

    ros::waitForShutdown();

    return 0;
}
