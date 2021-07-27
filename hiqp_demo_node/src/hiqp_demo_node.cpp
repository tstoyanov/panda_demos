#include <hiqp_demo_node/hiqp_demo_node.h>

using namespace hiqp_panda_demo; 

//-------------------------- Grasp Interval --------------------------------//
void GraspInterval::setInterval(std::string obj_frame, std::string e_frame, Eigen::Vector3d &eef,
		    Eigen::Affine3d &obj_pose) {
  //members
  obj_frame_ = obj_frame;  // object frame
  e_frame_ = e_frame;      // endeffector frame
  e_ = eef;                // end effector offset

  Eigen::Vector3d obj_t = obj_pose.translation();

  double obj_h = 0.05;
  double obj_r = 0.03;
  double obj_r_tol = 0.01;

  //------------Primitives---------------//
  //end effector point
  point_eef = hiqp_ros::createPrimitiveMsg(
      "point_eef", "point", e_frame_, false, {0, 0, 1, 0.2},
      {e_(0), e_(1), e_(2)});
  upper = hiqp_ros::createPrimitiveMsg(
      "upper", "plane", obj_frame_, false, {0, 0, 1, 0.2},
      {0, 0, -1, -obj_h});
  lower = hiqp_ros::createPrimitiveMsg(
      "lower", "plane", obj_frame_, false, {0, 0, 1, 0.2},
      {0, 0, 1, 0.0});
  inner = hiqp_ros::createPrimitiveMsg(
      "inner", "cylinder", obj_frame_, false, {0.6, 0, 0.6, 0.2},
      {0,0,1, obj_t(0),obj_t(1),obj_t(2), obj_r, obj_h}); 
  //principal axis direction, point, radius, height
  outer = hiqp_ros::createPrimitiveMsg(
      "outer", "cylinder", obj_frame_, false, {0.6, 0, 0.6, 0.2},
      {0,0,1, obj_t(0),obj_t(1),obj_t(2), obj_r+obj_r_tol, obj_h}); 
  //principal axis direction, point, radius, height
  
  obj_z_axis = hiqp_ros::createPrimitiveMsg(
      "obj_z_axis", "line", obj_frame_, false, {0.6, 0, 0.6, 0.2},
      {0,0,1, obj_t(0),obj_t(1),obj_t(2)}); 
  eef_approach_axis = hiqp_ros::createPrimitiveMsg(
      "eef_approach_axis", "line", e_frame_, false, {0.6, 0, 0.6, 0.2},
      {0,0,1, 0,0,0}); 
  eef_orthogonal_axis = hiqp_ros::createPrimitiveMsg(
      "eef_orthogonal_axis", "line", e_frame_, false, {0.6, 0, 0.6, 0.2},
      {1,0,0, 0,0,0}); 



  //------------Tasks-------------//
  // Upper/lower GRASP INTERVAL PLANE
  tasks_.push_back(hiqp_ros::createTaskMsg(
      "upperTask", 1, false, true, true,
      {"TDefGeomProj", "point", "plane",
       point_eef.name + " > " + upper.name},
      {"TDynLinear", std::to_string(2.0 * DYNAMICS_GAIN)}));
  tasks_.push_back(hiqp_ros::createTaskMsg(
      "lowerTask", 1, false, true, true,
      {"TDefGeomProj", "point", "plane",
       point_eef.name + " > " + lower.name},
      {"TDynLinear", std::to_string(2.0 * DYNAMICS_GAIN)}));
  tasks_.push_back(hiqp_ros::createTaskMsg(
      "innerTask", 2, false, true, true,
      {"TDefGeomProj", "point", "cylinder",
       point_eef.name + " > " + inner.name},
      {"TDynLinear", std::to_string(2.0 * DYNAMICS_GAIN)}));
  tasks_.push_back(hiqp_ros::createTaskMsg(
      "outerTask", 2, false, true, true,
      {"TDefGeomProj", "point", "cylinder",
       point_eef.name + " < " + outer.name},
      {"TDynLinear", std::to_string(2.0 * DYNAMICS_GAIN)}));

  tasks_.push_back(hiqp_ros::createTaskMsg(
      "orth_align", 3, false, true, true,
      {"TDefGeomAlign", "line", "line",
       eef_orthogonal_axis.name + " = " + obj_z_axis.name, "0.1"},
      {"TDynLinear", std::to_string(2.0 * DYNAMICS_GAIN)}));
  tasks_.push_back(hiqp_ros::createTaskMsg(
      "approach_proj", 3, false, true, true,
      {"TDefGeomProj", "line", "line",
       eef_approach_axis.name + " = " + obj_z_axis.name},
      {"TDynLinear", std::to_string(2.0 * DYNAMICS_GAIN)}));

}

void GraspInterval::getPrimitiveList(std::vector<hiqp_msgs::Primitive> &primitives) {
  if(!initialized) return;
  primitives.clear();
  primitives.push_back(point_eef);
  primitives.push_back(upper);
  primitives.push_back(lower);
  //primitives.push_back(left);
  //primitives.push_back(right);
  primitives.push_back(inner);
  primitives.push_back(outer);
  primitives.push_back(obj_z_axis);
  primitives.push_back(eef_approach_axis);
  primitives.push_back(eef_orthogonal_axis);
}

void GraspInterval::getTasksList(std::vector<hiqp_msgs::Task> &tasks) {
  if(!initialized) return;
  tasks.clear();
  for(auto it=tasks_.begin(); it!=tasks_.end(); it++) {
    tasks.push_back(*it);
  }
}
    
void GraspInterval::getPrimitiveNames(std::vector<std::string> &primitives) {
  if(!initialized) return;
  primitives.clear();
  primitives.push_back(point_eef.name);
  primitives.push_back(upper.name);
  primitives.push_back(lower.name);
  primitives.push_back(left.name);
  primitives.push_back(right.name);
  primitives.push_back(inner.name);
  primitives.push_back(outer.name);
  primitives.push_back(obj_z_axis.name);
  primitives.push_back(eef_approach_axis.name);
  primitives.push_back(eef_orthogonal_axis.name);

}

void GraspInterval::getTaskNames(std::vector<std::string> &tasks) {
  if(!initialized) return;
  tasks.clear();
  for(auto it=tasks_.begin(); it!=tasks_.end(); it++) {
    tasks.push_back(it->name);
  }

}

//-------------------------- Top Grasp Interval --------------------------//
void TopGraspInterval::setInterval(std::string obj_frame, std::string e_frame, Eigen::Vector3d &eef,
		    Eigen::Affine3d &obj_pose) {

}

void TopGraspInterval::getPrimitiveList(std::vector<hiqp_msgs::Primitive> &primitives) {
  if(!initialized) return;
  primitives.clear();
  primitives.push_back(cylinder);
  primitives.push_back(vertical_axis);
  primitives.push_back(horizontal_axis);
  primitives.push_back(plane_above);
  primitives.push_back(plane_below);   
}

void TopGraspInterval::getTasksList(std::vector<hiqp_msgs::Task> &tasks) {
  if(!initialized) return;
  tasks.clear();
  for(auto it=tasks_.begin(); it!=tasks_.end(); it++) {
    tasks.push_back(*it);
  }
}
    
void TopGraspInterval::getPrimitiveNames(std::vector<std::string> &primitives) {

}

void TopGraspInterval::getTaskNames(std::vector<std::string> &tasks) {

}


//Constructor, creates handles and gets stuff off the parameter server
DemoGrasping::DemoGrasping(): n_jnts(7), hiqp_client_("", "hiqp_joint_velocity_controller") {
  // handle to home
  nh_ = ros::NodeHandle("~");
  // global handle
  n_ = ros::NodeHandle();

  // register general callbacks
  start_demo_srv_ =
      nh_.advertiseService("start_demo", &DemoGrasping::startDemo, this);
  
  double px,py,pz;
  nh_.param<double>("eef_x", px, 0.0);
  nh_.param<double>("eef_y", py, 0.0);
  nh_.param<double>("eef_z", pz, 0.1);
  eef_offset_<<px,py,pz;

  nh_.param<double>("obj_r", px, 0.0);
  nh_.param<double>("obj_p", py, 0.0);
  nh_.param<double>("obj_y", pz, 0.0);
  object_pose_ = Eigen::AngleAxisd(pz, Eigen::Vector3d::UnitZ()) * 
                       Eigen::AngleAxisd(py, Eigen::Vector3d::UnitY()) * 
                       Eigen::AngleAxisd(px, Eigen::Vector3d::UnitX());
  
  nh_.param<double>("obj_x", px, 0.2);
  nh_.param<double>("obj_y", py, 0.4);
  nh_.param<double>("obj_z", pz, 0.0);
  object_pose_.translation()<<px,py,pz;


  std::cerr<<"Object pose is "<<object_pose_.matrix()<<std::endl;
  start_config_ = {1.5, -1.2, 0, -2.8, 0, 1.3, 0.8};

}

//starts up the demo execution. **BLOCKING service**
bool DemoGrasping::startDemo(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res) {

  //set initial configuration. 
  double tol = 1e-3;
  std::vector<std::string> def_params{"TDefFullPose"};

  for (auto jointValue : start_config_) {
    def_params.push_back(std::to_string(jointValue));
  }

  ROS_INFO("Setting initial configuration");
  bool ret = hiqp_client_.setTask("joint_configuration", 1, true, true, true, def_params,
                {"TDynLinear", "0.75"});
  if (ret) {
     hiqp_client_.waitForCompletion({"joint_configuration"}, {hiqp_ros::TaskDoneReaction::REMOVE},
                  {tol});
  } else {
      ROS_ERROR("could not set joint configuration task");
  }

  ROS_INFO("Done. Now setting picking tasks");
  //setup picking task
  GraspInterval grasp_interval_;
  grasp_interval_.setInterval("panda_link0", "panda_hand", eef_offset_, object_pose_);

  std::vector<hiqp_msgs::Primitive> grasp_primitives;
  grasp_interval_.getPrimitiveList(grasp_primitives);

  std::vector<hiqp_msgs::Task> grasp_tasks;
  grasp_interval_.getTasksList(grasp_tasks);

  hiqp_client_.setPrimitives(grasp_primitives);
  hiqp_client_.setTasks(grasp_tasks);
  
  std::vector<std::string> task_names;
  std::vector<double> tolerances;
  std::vector<hiqp_ros::TaskDoneReaction> reactions;
  for(auto it=grasp_tasks.begin(); it!=grasp_tasks.end(); it++) {
	  task_names.push_back(it->name);
	  tolerances.push_back(tol);
	  reactions.push_back(hiqp_ros::TaskDoneReaction::REMOVE);
  }

  ROS_INFO("Waiting for completion");
  hiqp_client_.waitForCompletion(task_names, reactions, tolerances);

  ROS_INFO("Done. Remove primitives.");
  std::vector<std::string> prim_names;
  grasp_interval_.getPrimitiveNames(prim_names);
  hiqp_client_.removePrimitives(prim_names);

  return true;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "demo_node");

  DemoGrasping demo_node;

  ROS_INFO("Demo node ready");
  ros::AsyncSpinner spinner(4);  // Use 4 threads
  spinner.start();
  ros::waitForShutdown();

  return 0;
}
