#include <ros/ros.h>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/path_roundedcomposite.hpp>
#include <kdl/rotational_interpolation_sa.hpp>
#include <kdl/chain.hpp>
#include <kdl/jntarray.hpp>

#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_wdls.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <boost/filesystem.hpp>

#include <ros/package.h>
#include <jsoncpp/json/json.h>
#include <tinyxml.h>

#include <math.h>
#include <random>

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

  // ===== getting joint limits from URDF model =====
  double average;
  TiXmlDocument tiny_doc;
  tiny_doc.Parse(robot_desc_string.c_str());
  std::map<std::string, std::map<std::string, double>> joint_limits;

  TiXmlHandle doc_handle {&tiny_doc};
  std::string joint_name;
  std::vector<std::string>::iterator it;
  for (TiXmlElement* tiny_joint = doc_handle.FirstChild("robot").Child("joint", 0).ToElement(); tiny_joint; tiny_joint = tiny_joint -> NextSiblingElement("joint"))
  {
    joint_name = tiny_joint -> Attribute("name");
    it = std::find(joint_names.begin(), joint_names.end(), joint_name);
    if (it != joint_names.end())
    {
      joint_limits[joint_name].insert({{"lower", std::stod(tiny_joint -> FirstChild("limit") -> ToElement() -> Attribute("lower"))}});
      joint_limits[joint_name].insert({{"upper", std::stod(tiny_joint -> FirstChild("limit") -> ToElement() -> Attribute("upper"))}});
    }
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

  KDL::Chain my_chain = KDL::Chain {};
  std::vector<std::string> chain_segments_names;
  my_tree.getChain("world", "panda_hand", my_chain);
  unsigned nr_of_joints = my_chain.getNrOfJoints();
  // std::cout << "my_chain.getNrOfJoints(): " << my_chain.getNrOfJoints() << std::endl;
  // for (unsigned i = 0; i < my_chain.getNrOfSegments(); i++)
  // {
  //   chain_segments_names.push_back(my_chain.getSegment(i).getName());
  //   std::cout << "\tchain segment" << i << ": " << my_chain.getSegment(i).getName() << std::endl;
  // }

  KDL::JntArray q_min {(unsigned) joint_names.size()};
  KDL::JntArray q_max {(unsigned) joint_names.size()};
  for (unsigned i = 0; i < joint_names.size(); i++)
  {
    q_min(i) = joint_limits[joint_names[i]]["lower"];
    q_max(i) = joint_limits[joint_names[i]]["upper"];
    // std::cout << "Joint " << joint_names[i] << "\n";
    // std::cout << "\tlower " << q_min(i) << "\n";
    // std::cout << "\tupper " << q_max(i) << "\n";
    // std::cout << "\taverage " << (q_min(i) + q_max(i)) / 2 << "\n";
  }
  unsigned max_iter = 100;
  double eps = 1e-12;

   // ==================== CHAIN SOLVER ====================
  KDL::ChainFkSolverPos_recursive chainFkSolverPos {my_chain};
  KDL::ChainIkSolverVel_wdls chainIkSolverVel {my_chain};
  KDL::ChainIkSolverPos_NR_JL chainIkSolverPos {my_chain, q_min, q_max, chainFkSolverPos, chainIkSolverVel, max_iter, eps};

  // ========== WAYPOINTS ==========
  std::vector<std::vector<double>> starting_waypoints =
  {
    // good
    // {x, y, z}
    {-0.401718997062, 0.0892002648095, 0.986710060669},
    {0.368281002938, 0.0892002648095, 0.886710060669},
    {0.468281002938, 0.0892002648095, 0.986710060669}
    
    // collision test
    // {x, y, z}
    // {-0.401718997062, 0.4892002648095, 0.986710060669},
    // {0.368281002938, 0.4892002648095, 0.886710060669},
    // {0.468281002938, 0.4892002648095, 0.986710060669}
    
    // old
    // {x, y, z}
    // {-0.531718997062, 0.0892002648095, 1.08671006067},
    // {-0.131718997062, 0.0892002648095, 0.886710060669},
    // {0.368281002938, 0.0892002648095, 0.886710060669},
    // {0.468281002938, 0.0892002648095, 0.986710060669}
  };

  KDL::RotationalInterpolation_SingleAxis * orient = new KDL::RotationalInterpolation_SingleAxis();
  KDL::Path_RoundedComposite path(radius, eqradius, orient, true);

  // ========== adding noise to the waypoints ==========
  // int sign
  bool path_error;
  double noise;
  std::default_random_engine generator {std::random_device()()};
  std::normal_distribution<double> distribution(0.0, 1.66); // 99% of the noise falls inside 3*stddev (1.66 = +/- 5 m?)
  
  // do
  // {
    path_error = false;
    std::vector<std::vector<double>> noisy_waypoints = starting_waypoints;
    for (unsigned i = 0; i < noisy_waypoints.size(); i++)
    {
      for (unsigned ii = 0; ii < noisy_waypoints[i].size(); ii++)
      {
        noise = distribution(generator) / 100.0;
        if (!(noise < 0.05 && noise > -0.05)){
          std::cout << "fail" << std::endl;
          noise = distribution(generator) / 100.0;
        }
        else
        {
          noisy_waypoints[i][ii] += noise;
        }
        std::cout << "\nnoise: " << noise << std::endl;
        std::cout << "noisy_waypoints[" << i << "][" << ii << "]: " << noisy_waypoints[i][ii] << std::endl;
      }
      path.Add(KDL::Frame(KDL::Vector(noisy_waypoints[i][0], noisy_waypoints[i][1], noisy_waypoints[i][2])));
    }
    // try
    // {
      path.Finish();
  //   }
  //   catch(const std::exception& e)
  //   {
  //     path_error = true;
  //     std::cout << "\nEXCEPTION\n";
  //     std::cerr << e.what() << '\n';
  //   }
  // }
  // while (!path_error);

  double current_s = 0;
  KDL::Frame current_eef_frame;
  KDL::Frame fk_current_eef_frame;
  std::vector<KDL::Frame> eef_trajectory;
  std::vector<KDL::Frame> fk_eef_trajectory;
  KDL::Vector current_eef_pos;
  KDL::Vector fk_current_eef_pos;
  std::vector<KDL::JntArray> joint_trajectory;
  KDL::JntArray q_out {nr_of_joints};
  KDL::JntArray last_joint_pos {nr_of_joints};

  int ret;
  double number_of_samples = 100;
  double path_length = path.PathLength();
  double ds = path_length / number_of_samples;
  double start_joint_pos_array[] = {-0.448125769162, 0.32964587676, -0.621680615641, -1.82515059054, 0.269715026327, 2.11438395741, -1.94274845254};
  double average_start_joint_pos_array[] = {0, 0, 0, -1.5708, 0, 1.8675, 0};
  double kdl_start_joint_pos_array[] = {-0.443379, 0.702188, -0.556869, -1.9368, -2.55769, 0.667764, -2.56121};

  for (unsigned i = 0; i < nr_of_joints; i++)
  {
    last_joint_pos(i) = start_joint_pos_array[i];
  }

  ret = chainFkSolverPos.JntToCart(last_joint_pos, fk_current_eef_frame);
  std::cout << "FK RET: " << ret << std::endl;
  KDL::Rotation starting_orientation = fk_current_eef_frame.M;

  // calculate the new joint coordinates for the starting point
  current_eef_frame = path.Pos(0);
  current_eef_frame.M = starting_orientation;
  ret = chainIkSolverPos.CartToJnt(last_joint_pos, current_eef_frame, last_joint_pos);
  std::cout << "New starting joint coordinates RET = " << ret << std::endl;

  for (unsigned i = 0; i < last_joint_pos.rows(); i++)
  {
    std::cout << "last_joint_pos(" << i << "): " << last_joint_pos(i) << std::endl;
  }
  joint_trajectory.push_back(last_joint_pos);
  eef_trajectory.push_back(path.Pos(0));

  // char c;
  // std::cin >> c;

  // ====================FK====================
  ret = chainFkSolverPos.JntToCart(last_joint_pos, fk_current_eef_frame);
  std::cout << "FK RET: " << ret << std::endl;
  fk_current_eef_pos = fk_current_eef_frame.p;
  fk_eef_trajectory.push_back(fk_current_eef_frame);
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
    std::cout << "EEF frame orientation" << std::endl;
    std::cout << "\t\t\tX: " << X << std::endl;
    std::cout << "\t\t\tY: " << Y << std::endl;
    std::cout << "\t\t\tZ: " << Z << std::endl;

    ret = chainIkSolverPos.CartToJnt(last_joint_pos, current_eef_frame, q_out);
    std::cout << "RET TRUE: " << ret << std::endl;
    joint_trajectory.push_back(q_out);

    ret = chainFkSolverPos.JntToCart(q_out, fk_current_eef_frame);
    std::cout << "------------------------------\n";
    std::cout << "FK RET: " << ret << std::endl;
    std::cout << "FK EEF pos: " << std::endl;
    std::cout << "\tx: " << fk_current_eef_frame.p.x() << std::endl;
    std::cout << "\ty: " << fk_current_eef_frame.p.y() << std::endl;
    std::cout << "\tz: " << fk_current_eef_frame.p.z() << std::endl;
    std::cout << "------------------------------\n";
    fk_eef_trajectory.push_back(fk_current_eef_frame);

    std::cout << "------------------------------\n";
    std::cout << "Point " << i << ": " << std::endl;
    std::cout << "x: " << current_eef_frame.p.x() << std::endl;
    std::cout << "y: " << current_eef_frame.p.y() << std::endl;
    std::cout << "z: " << current_eef_frame.p.z() << std::endl;
    std::cout << "q_out:\n" << q_out.data << std::endl;
    std::cout << "------------------------------\n";
    last_joint_pos = q_out;
  }

  // ====================JSON====================
  Json::Value data;
  for (unsigned i = 0; i < joint_names.size(); i++)
  {
    data["joint_names"].append(Json::Value(joint_names[i]));
  }
  for (unsigned i = 0; i < joint_trajectory.size(); i++)
  {
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
  std::string latest_dir_path = pkg_path + "/generated_trajectories/cpp/latest";
  std::string dir_path = pkg_path + "/generated_trajectories/cpp/" + std::to_string(ts);
  boost::filesystem::path dir(dir_path);
  boost::filesystem::path latest_dir(latest_dir_path);
  if(!(boost::filesystem::exists(dir)))
  {
    if (boost::filesystem::create_directories(dir))
    {
      std::cout << "....'timestamp' folder Successfully Created!" << std::endl;
    }
    else
    {
      std::cout << "....ERROR 'timestamp' folder Couldn't Be Created!" << std::endl;
    }
  }
  if(!(boost::filesystem::exists(latest_dir)))
  {
    if (boost::filesystem::create_directories(latest_dir))
    {
      std::cout << "....'latest' folder Successfully Created!" << std::endl;
    }
    else
    {
      std::cout << "....ERROR 'latest' folder Couldn't Be Created!" << std::endl;
    }
  }
  std::ofstream myfile (dir_path + "/trajectories.txt");
  if (myfile.is_open())
  {
    myfile << data << std::endl;
    myfile.close();
  }

  myfile.open(latest_dir_path + "/trajectories.txt");
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

  return 0;
}
