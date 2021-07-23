#include <hiqp_demo_node/hiqp_demo_node.h>

using namespace hiqp_panda_demo; 

void GraspInterval::setInterval() {

}

void GraspInterval::getPrimitiveList(std::vector<hiqp_msgs::Primitive> &primitives) {
  if(!initialized) return;
  primitives.clear();
  primitives.push_back(upper);
  primitives.push_back(lower);
  primitives.push_back(left);
  primitives.push_back(right);
  primitives.push_back(inner);
  primitives.push_back(outer);
}

void GraspInterval::getTasksList(std::vector<hiqp_msgs::Task> &tasks) {
  if(!initialized) return;
  tasks.clear();
  for(auto it=tasks_.begin(); it!=tasks_.end(); it++) {
    tasks.push_back(*it);
  }
}
    
void GraspInterval::getPrimitiveNames(std::vector<std::string> &primitives) {

}

void GraspInterval::getTaskNames(std::vector<std::string> &tasks) {

}

//Constructor, creates handles and gets stuff off the parameter server
DemoNode::DemoNode(): n_jnts(7), hiqp_client_("", "hiqp_joint_velocity_controller") {
  // handle to home
  nh_ = ros::NodeHandle("~");
  // global handle
  n_ = ros::NodeHandle();
}

//starts up the demo execution. **BLOCKING service**
bool DemoNode::startDemo(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res) {

  return true;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "demo_node");

  DemoNode demo_node;

  ROS_INFO("Demo node ready");
  ros::AsyncSpinner spinner(4);  // Use 4 threads
  spinner.start();
  ros::waitForShutdown();

  return 0;
}
