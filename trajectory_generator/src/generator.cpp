#include <ros/ros.h>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/path_roundedcomposite.hpp>
#include <kdl/rotational_interpolation_sa.hpp>
#include <kdl/treeiksolverpos_nr_jl.hpp>
#include <kdl/chain.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/treefksolverpos_recursive.hpp>
#include <kdl/treeiksolvervel_wdls.hpp>

#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_wdls.hpp>

#include <kdl/chainiksolvervel_pinv_givens.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainiksolverpos_nr.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <boost/filesystem.hpp>

#include <ros/package.h>

#include <json.hpp>
using json = nlohmann::json;
#include <pugixml.hpp>

#include <boost/math/constants/constants.hpp>
const double pi = boost::math::constants::pi<double>();

std::string getExePath()
{
  char result[ PATH_MAX ];
  ssize_t count = readlink( "/proc/self/exe", result, PATH_MAX );
  return std::string( result, (count > 0) ? count : 0 );
}

int main(int argc, char **argv)
{
  double radius = 0.3;
  double eqradius = 0.3;

  if (argc > 1) 
  {
    radius = std::stod(argv[1]);
    eqradius = std::stod(argv[1]);
  }
  ros::init(argc, argv, "myGenerator");
  KDL::Tree my_tree {};
  ros::NodeHandle node;
  ros::Rate loop_rate(2);

  // ==========REAL POINTS==========
  double x0 = -0.531718997062;
  double y0 = 0.0892002648095;
  double z0 = 1.08671006067;

  double x1 = -0.131718997062;
  double y1 = 0.0892002648095;
  double z1 = 0.886710060669;

  double x2 = 0.368281002938;
  double y2 = 0.0892002648095;
  double z2 = 0.886710060669;

  double x3 = 0.468281002938;
  double y3 = 0.0892002648095;
  double z3 = 0.986710060669;

  KDL::Vector position_vector_0 {x0, y0, z0};
  KDL::Frame waypoint_frame_0 {position_vector_0};
  KDL::Vector position_vector_1 {x1, y1, z1};
  KDL::Frame waypoint_frame_1 {position_vector_1};
  KDL::Vector position_vector_2 {x2, y2, z2};
  KDL::Frame waypoint_frame_2 {position_vector_2};
  KDL::Vector position_vector_3 {x3, y3, z3};
  KDL::Frame waypoint_frame_3 {position_vector_3};

  KDL::RotationalInterpolation_SingleAxis * orient = new KDL::RotationalInterpolation_SingleAxis();

  KDL::Path_RoundedComposite path(radius, eqradius, orient, true);

  path.Add(waypoint_frame_0);
  path.Add(waypoint_frame_1);
  path.Add(waypoint_frame_2);
  path.Add(waypoint_frame_3);
  path.Finish();

  // ===== getting URDF model form param server =====
  std::string robot_desc_string;
  node.param("robot_description", robot_desc_string, std::string());
  if (!kdl_parser::treeFromString(robot_desc_string, my_tree)){
    ROS_ERROR("Failed to construct kdl tree");
    return false;
  }

  // ===== getting joint names from param server =====
  std::string param_server_joints = "/position_joint_trajectory_controller/joints";
  std::vector<std::string> joint_names;
  node.getParam(param_server_joints, joint_names);
  for (unsigned i = 0; i < joint_names.size(); i++)
  {
    std::cout << "joint names: " << joint_names[i] << "\n";
  }

  // ===== getting joint limits from URDF model =====
  pugi::xml_document doc;
  pugi::xml_node limit;
  pugi::xml_parse_result result = doc.load_string(robot_desc_string.c_str());
  // std::cout << "Load result: " << result.description() << ", robot name: " << doc.child("robot").attribute("name").value() << std::endl;

  double average;
  std::string joint_name;
  std::map<std::string, std::map<std::string, double>> joint_limits;
  for (pugi::xml_node joint = doc.child("robot").child("joint"); joint; joint = joint.next_sibling("joint"))
  {
    joint_name = joint.attribute("name").value();
    if (std::find(joint_names.begin(), joint_names.end(), joint_name) != joint_names.end())
    {
      limit = joint.child("limit");
      for (pugi::xml_attribute_iterator ait = limit.attributes_begin(); ait != limit.attributes_end(); ++ait)
      {
        // std::cout << "\t" << ait->name() << "=" << ait->value() << std::endl;
        joint_limits[joint_name].insert({{ait->name(), std::stod(ait->value())}});
      }
      
      // ====================FAKE LIMITS====================
      // joint_limits["panda_joint1"]["lower"] = -0.6;
      // joint_limits["panda_joint1"]["upper"] = 1.0;
      // joint_limits["panda_joint2"]["lower"] = -0.1;
      // joint_limits["panda_joint2"]["upper"] = 0.4;
      // joint_limits["panda_joint3"]["lower"] = -0.7;
      // joint_limits["panda_joint3"]["upper"] = 0.25;
      // joint_limits["panda_joint4"]["lower"] = -3.0;
      // joint_limits["panda_joint4"]["upper"] = -1.5;
      // joint_limits["panda_joint5"]["lower"] = -0.25;
      // joint_limits["panda_joint5"]["upper"] = 0.3;
      // joint_limits["panda_joint6"]["lower"] = 1.8;
      // joint_limits["panda_joint6"]["upper"] = 3.25;
      // joint_limits["panda_joint7"]["lower"] = -2.2;
      // joint_limits["panda_joint7"]["upper"] = 0.4;
      // ====================END FAKE LIMITS====================
    }
  }

  KDL::Chain my_chain = KDL::Chain {};
  std::vector<std::string> chain_segments_names;
  my_tree.getChain("table", "panda_hand", my_chain);
  unsigned int nr_of_joints = my_chain.getNrOfJoints();
  // my_tree.getChain("mount_flange_link", "panda_hand", my_chain);
  // std::cout << "my_chain.getNrOfJoints(): " << my_chain.getNrOfJoints() << std::endl;
  // for (unsigned i = 0; i < my_chain.getNrOfSegments(); i++)
  // {
  //   chain_segments_names.push_back(my_chain.getSegment(i).getName());
  //   std::cout << "\tchain segment" << i << ": " << my_chain.getSegment(i).getName() << std::endl;
  // }

  KDL::JntArray q_min {(unsigned int) joint_names.size()};
  KDL::JntArray q_max {(unsigned int) joint_names.size()};
  for (unsigned i = 0; i < joint_names.size(); i++)
  {
    q_min(i) = joint_limits[joint_names[i]]["lower"];
    q_max(i) = joint_limits[joint_names[i]]["upper"];
    
    std::cout << "Joint " << joint_names[i] << "\n";
    std::cout << "\tlower " << q_min(i) << "\n";
    std::cout << "\tupper " << q_max(i) << "\n";
    std::cout << "\taverage " << (q_min(i) + q_max(i)) / 2 << "\n";
  }
  unsigned int max_iter = 100;
  double eps = 1e-12;

  double number_of_samples = 100;
  double path_length = path.PathLength();
  double ds = path_length / number_of_samples;
  double start_joint_pos_array[] = {-0.448125769162, 0.32964587676, -0.621680615641, -1.82515059054, 0.269715026327, 2.11438395741, -1.94274845254};
  double average_start_joint_pos_array[] = {0, 0, 0, -1.5708, 0, 1.8675, 0};
  double kdl_start_joint_pos_array[] = {-0.443379, 0.702188, -0.556869, -1.9368, -2.55769, 0.667764, -2.56121};

  double current_s = 0;
  KDL::Frame current_eef_frame;
  KDL::Frame fk_current_eef_frame;
  KDL::Frames current_frames;
  std::vector<KDL::Frame> eef_trajectory;
  std::vector<KDL::Frame> fk_eef_trajectory;
  KDL::Vector current_eef_pos;
  KDL::Vector fk_current_eef_pos;
  std::vector<KDL::JntArray> joint_trajectory;
  KDL::JntArray q_out {nr_of_joints};
  KDL::JntArray q_out_norm {nr_of_joints};

  KDL::JntArray last_joint_pos {nr_of_joints};
  for (unsigned i = 0; i < nr_of_joints; i++)
  {
    // last_joint_pos(i) = average_start_joint_pos_array[i];
    last_joint_pos(i) = start_joint_pos_array[i];
  }
  joint_trajectory.push_back(last_joint_pos);
  eef_trajectory.push_back(path.Pos(0));
  
  // ====================CHAIN SOLVER====================
  KDL::ChainFkSolverPos_recursive chainFkSolverPos {my_chain};
  KDL::ChainIkSolverVel_wdls chainIkSolverVel {my_chain};
  KDL::ChainIkSolverPos_NR_JL chainIkSolverPos {my_chain, q_min, q_max, chainFkSolverPos, chainIkSolverVel, max_iter, eps};

  int ret;

  // ====================FK====================
  ret = chainFkSolverPos.JntToCart(last_joint_pos, fk_current_eef_frame);
  std::cout << "FK RET: " << ret << std::endl;
  fk_current_eef_pos = fk_current_eef_frame.p;
  fk_eef_trajectory.push_back(fk_current_eef_frame);
  KDL::Rotation starting_orientation = fk_current_eef_frame.M;
  std::cout << "------------------------------\n";
  std::cout << "FK EEF pos: " << std::endl;
  std::cout << "\tx: " << fk_current_eef_pos.x() << std::endl;
  std::cout << "\ty: " << fk_current_eef_pos.y() << std::endl;
  std::cout << "\tz: " << fk_current_eef_pos.z() << std::endl;
  std::cout << "------------------------------\n";
  // ====================END FK====================

  for (unsigned i = 0; i < number_of_samples; i++)
  {
    double X;
    double Y;
    double Z;
    double W;

    current_s += ds;
    current_eef_frame = path.Pos(current_s);
    current_eef_frame.M = starting_orientation;
    eef_trajectory.push_back(current_eef_frame);

    current_eef_frame.M.GetEulerZYX(Z, Y, X);
    std::cout << "\t\t\tX: " << X << std::endl;
    std::cout << "\t\t\tY: " << Y << std::endl;
    std::cout << "\t\t\tZ: " << Z << std::endl;

    ret = chainIkSolverPos.CartToJnt(last_joint_pos, current_eef_frame, q_out);
    std::cout << "RET TRUE: " << ret << std::endl;
    q_out_norm = q_out;
    for (unsigned n = 0; n < nr_of_joints; n++)
    {
      // std::cout << "--------------------------\nbefore: " << n << "\n" << q_out.data[n] << std::endl;
      // q_out_norm.data[n] = q_out.data[n] * pi / 180;z
      // q_out_norm.data[n] = fmod(q_out.data[n], pi);
      // std::cout << "after:\n" << q_out_norm.data[n] << "\n--------------------------" <<std::endl;
    }
    joint_trajectory.push_back(q_out_norm);
    // joint_trajectory.push_back(q_out);

    ret = chainFkSolverPos.JntToCart(q_out_norm, fk_current_eef_frame);
    // ret = chainFkSolverPos.JntToCart(q_out, fk_current_eef_frame);
    std::cout << "------------------------------\n";
    std::cout << "FK RET: " << ret << std::endl;
    std::cout << "FK EEF pos: " << std::endl;
    std::cout << "\tx: " << fk_current_eef_frame.p.x() << std::endl;
    std::cout << "\ty: " << fk_current_eef_frame.p.y() << std::endl;
    std::cout << "\tz: " << fk_current_eef_frame.p.z() << std::endl;
    std::cout << "------------------------------\n";
    fk_eef_trajectory.push_back(fk_current_eef_frame);

    // std::cout << "\nret: " << ret << std::endl;
    // std::cout << "my_chain.getNrOfJoints(): " << my_chain.getNrOfJoints() << std::endl;

    std::cout << "------------------------------\n";
    std::cout << "Point " << i << ": " << std::endl;
    std::cout << "x: " << current_eef_frame.p.x() << std::endl;
    std::cout << "y: " << current_eef_frame.p.y() << std::endl;
    std::cout << "z: " << current_eef_frame.p.z() << std::endl;
    std::cout << "q_out:\n" << q_out.data << std::endl;
    std::cout << "q_out_norm:\n" << q_out_norm.data << std::endl;
    std::cout << "------------------------------\n";
    last_joint_pos = q_out;
  }

  // ====================JSON====================
  json data;
  for (unsigned i = 0; i < joint_trajectory.size(); i++)
  {
    data["joint_names"] = joint_names;
    for (unsigned ii = 0; ii < nr_of_joints; ii++)
    {
      data["joint_trajectory"][i][ii] = joint_trajectory[i].data[ii];
    }

    data["eef_trajectory"][i]["origin"]["x"] = eef_trajectory[i].p.x();
    data["eef_trajectory"][i]["origin"]["y"] = eef_trajectory[i].p.y();
    data["eef_trajectory"][i]["origin"]["z"] = eef_trajectory[i].p.z();

    data["fk_eef_trajectory"][i]["origin"]["x"] = fk_eef_trajectory[i].p.x();
    data["fk_eef_trajectory"][i]["origin"]["y"] = fk_eef_trajectory[i].p.y();
    data["fk_eef_trajectory"][i]["origin"]["z"] = fk_eef_trajectory[i].p.z();
  }

  std::string pkg_path = ros::package::getPath("trajectory_generator");
  std::cout << "pkg_path: " << pkg_path << std::endl;
  long long ts = std::chrono::system_clock::now().time_since_epoch().count();
  std::cout << "time: " << ts << std::endl;
  std::string dir_path = pkg_path + "/generated_trajectories/cpp/" + std::to_string(ts);
  boost::filesystem::path dir(dir_path);
  if(!(boost::filesystem::exists(dir)))
  {
    if (boost::filesystem::create_directory(dir))
    {
      std::cout << "....Folder Successfully Created!" << std::endl;
    }
    else
    {
      std::cout << "....ERROR Folder Couldn't Be Created!" << std::endl;
    }
  }
  std::ofstream myfile (dir_path + "/trajectories.txt");
  if (myfile.is_open())
  {
    myfile << data << std::endl;
    myfile.close();
  }
  // ====================END JSON====================
  
  int n = path.GetNrOfSegments();
  double length = path.GetLengthToEndOfSegment(n);
  std::cout << "my_tree.getNrOfJoints(): " << my_tree.getNrOfJoints() << std::endl;
  std::cout << "path.GetNrOfSegments(): " << path.GetNrOfSegments() << std::endl;
  std::cout << "path.GetLengthToEndOfSegment(" << path.GetNrOfSegments()-1 << "): " << path.GetLengthToEndOfSegment(n-1) << std::endl;
  std::cout << "path.PathLength(): " << path.PathLength() << std::endl;
  
  unsigned i = 0;
  while (ros::ok())
  {
    i++;
    if (i > 1) {
      ros::shutdown();
    }
  }

  return 0;
}
