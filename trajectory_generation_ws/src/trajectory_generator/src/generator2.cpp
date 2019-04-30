#include <ros/ros.h>
#include <kdl_parser/kdl_parser.hpp>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "myGenerator");
  ros::NodeHandle nh;
  ros::Rate loop_rate(2);

  // KDL::Tree my_tree;
  // std::string file_name = "/home/ilbetzy/orebro/trajectory_generation_ws/panda_table.urdf";
  // if (!kdl_parser::treeFromFile(file_name, my_tree))
  // {
  //   ROS_ERROR("Failed to construct kdl tree");
  //   return false;
  // }
  
  while (ros::ok())
  {
    ROS_INFO("test");
    // printf("my_tree: %s", my_tree);
  }

  return 0;
}
