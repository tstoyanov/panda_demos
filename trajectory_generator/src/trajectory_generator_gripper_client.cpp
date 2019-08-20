#include "ros/ros.h"
#include "trajectory_generator/trajectory_generator_gripper.h"
#include <cstdlib>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "trajectory_generator_client");
  if (argc != 3)
  {
    ROS_INFO("usage: trajectory_generator_client X Y");
    return 1;
  }

  ros::NodeHandle n;
  ros::ServiceClient client = n.serviceClient<trajectory_generator::trajectory_generator_gripper>("trajectory_generator_gripper");
  trajectory_generator::trajectory_generator_gripper srv;
  srv.request.a = atoll(argv[1]);
  srv.request.b = atoll(argv[2]);
  if (client.call(srv))
  {
    ROS_INFO("Sum: %ld", (long int)srv.response.sum);
  }
  else
  {
    ROS_ERROR("Failed to call service trajectory_generator_gripper");
    return 1;
  }

  return 0;
}