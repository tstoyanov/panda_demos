#include "ros/ros.h"
#include "trajectory_generator/trajectory_generator_gripper.h"

bool add(trajectory_generator::trajectory_generator_gripper::Request  &req,
         trajectory_generator::trajectory_generator_gripper::Response &res)
{
  res.sum = req.a + req.b;
  ROS_INFO("request: x=%ld, y=%ld", (long int)req.a, (long int)req.b);
  ROS_INFO("sending back response: [%ld]", (long int)res.sum);
  return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "trajectory_generator_gripper_server");
  ros::NodeHandle n;

  ros::ServiceServer service = n.advertiseService("trajectory_generator_gripper", add);
  ROS_INFO("Ready to add two ints.");
  ros::spin();

  return 0;
}