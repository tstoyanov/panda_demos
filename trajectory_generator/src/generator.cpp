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
#include <boost/program_options.hpp>

#include <ros/package.h>
#include <jsoncpp/json/json.h>
#include <tinyxml.h>

#include <math.h>
#include <random>

#include "velocity_profile_generator.cpp"

#define RADIUS 0.3
#define EQRADIUS 0.3

#define MAX_PATH_GENERATION_ATTEMPTS 100
#define MAX_NOISE_GENERATION_ATTEMPTS 100

#define NUMBER_OF_SAMPLES 100
#define RELEASE_FRAME 0.8 * NUMBER_OF_SAMPLES

#define BINARY_SEARCH_TRESHOLD 0.0001 // meters
#define MAX_NUMBER_OF_INVERSIONS 10

std::vector<std::map<std::string, double>> NOISE_MATRIX = { // 1 will be converted to 1 cm for now
  { // x                                                    // 99% of the noise falls inside MEAN +/- 3*NOISE_STDDEV (1.66 = +/- 5m? => 5cm in the end)
    {"noise_mean", 0}, {"noise_stddev", 3}
  },
  { // y
    {"noise_mean", 0}, {"noise_stddev", 0.3}
  },
  { // z
    {"noise_mean", 0}, {"noise_stddev", 2}
  }
};


double binary_search_treshold (KDL::Path_RoundedComposite &path, double lower_bound, double upper_bound, const double &x_value, const double &treshold)
{
  double mid = (upper_bound + lower_bound) / 2;
  double mid_x_value = path.Pos(mid).p.x();
  if (mid_x_value <= (x_value + treshold) && mid_x_value >= (x_value - treshold))
  {
    return mid;
  }
  else
  {
    if (mid_x_value < x_value)
    {
      return binary_search_treshold (path, mid, upper_bound, x_value, treshold);
    }
    else
    {
      return binary_search_treshold (path, lower_bound, mid, x_value, treshold);
    }
  }
}

double minima_search (KDL::Path_RoundedComposite &path, double ds, int max_number_of_inversions)
{
  int sign = 1;
  int inversions = 0;
  double current_z;
  double current_s = ds;
  double z_min = path.Pos(0).p.z();
  while (current_s <= path.PathLength() && current_s >= 0)
  {
    current_z = path.Pos(current_s).p.z();
    if (z_min > current_z)
    {
      z_min = current_z;
    }
    else
    {
      inversions++;
      if (inversions >= max_number_of_inversions)
      {
        return current_s;
      }
      // ds /= 2;
      // sign *= -1;
      ds = ds / 2;
      sign = sign * (-1);
    }
    current_s += sign * ds;
  }
}

std::string getExePath()
{
  char result[ PATH_MAX ];
  ssize_t count = readlink( "/proc/self/exe", result, PATH_MAX );
  return std::string( result, (count > 0) ? count : 0 );
}

int main(int argc, char **argv)
{
  bool is_batch = false;
  bool is_simulation = false;
  int batch_count = 1;
  int generated_trajectories = 0;
  namespace po = boost::program_options;
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("batch,b", po::value<int>(&batch_count), "set the number of trajectories to generate")
    ("sim,s", po::bool_switch()->default_value(false), "simulation flag");

  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  if (vm.count("batch"))
  {
    batch_count = vm["batch"].as<int>();
    if (batch_count >= 1)
    {
      is_batch = true;
      std::cout << "batch argument was set to: " << batch_count << std::endl;
    }
    else
    {
      batch_count = 1;
      std::cout << "batch argument was lesser than 1, a single trajectory will be generated" << std::endl;
    }
  }
  else
  {
    std::cout << "batch argument was not set, a single trajectory will be generated" << std::endl;
  }
  
  if (vm.count("sim"))
  {
    is_simulation = vm["sim"].as<bool>();
    std::cout << "SIMULATION MODE" << std::endl;
  }

  // if (argc > 1)
  // {
  //   RADIUS = std::stod(argv[1]);
  //   EQRADIUS = std::stod(argv[1]);
  // }
  ros::init(argc, argv, "myGenerator");
  KDL::Tree my_tree {};
  ros::NodeHandle node;
  ros::Rate loop_rate(2);
  

  // ===== getting URDF model and joint names form param server =====
  std::string robot_desc_string;
  std::vector<std::string> joint_names;
  if (is_simulation)
  {
    node.param("robot_description", robot_desc_string, std::string());
    node.getParam("position_joint_trajectory_controller/joints", joint_names);
  }
  else
  {
    node.param("/panda/robot_description", robot_desc_string, std::string());
    node.getParam("/panda/position_joint_trajectory_controller/joints", joint_names);
  }
  
  if (!kdl_parser::treeFromString(robot_desc_string, my_tree)){
    ROS_ERROR("Failed to construct kdl tree");
    return false;
  }

  // ===== getting joint names from param server =====
  // std::string param_server_joints = "/panda/position_joint_trajectory_controller/joints";
  // std::vector<std::string> joint_names;
  // node.getParam(param_server_joints, joint_names);

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

  double release_x_coordinate = 0.368281002938;
  double noisy_release_x_coordinate;
  // ========== WAYPOINTS ==========
  std::vector<std::vector<double>> starting_waypoints =
  {
    // good
    // {x, y, z}
    {-0.401718997062, 0.0892002648095, 0.986710060669},
    {release_x_coordinate, 0.0892002648095, 0.886710060669},
    {0.568281002938, 0.0892002648095, 1.086710060669}

    // {-0.401718997062, 0.0892002648095, 0.916710060669},
    // {release_x_coordinate, 0.0892002648095, 0.866710060669},
    // {0.568281002938, 0.0892002648095, 1.086710060669}
    
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
  KDL::Path_RoundedComposite *path;

  // ========== adding noise to the waypoints ==========
  bool path_error;
  double noise;
  int noise_generation_counter;
  std::vector<std::vector<double>> noisy_waypoints;
  double max_noise;
  double min_noise;
  std::default_random_engine generator {std::random_device()()};
  std::normal_distribution<double> distribution;
  int exception_count;

  double current_s;
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
  double path_length;
  double ds;
  double start_joint_pos_array[] = {-0.448125769162, 0.32964587676, -0.621680615641, -1.82515059054, 0.269715026327, 2.11438395741, -1.94274845254};
  double average_start_joint_pos_array[] = {0, 0, 0, -1.5708, 0, 1.8675, 0};
  double kdl_start_joint_pos_array[] = {-0.443379, 0.702188, -0.556869, -1.9368, -2.55769, 0.667764, -2.56121};

  double X;
  double Y;
  double Z;
  double W;

  KDL::Rotation starting_orientation;
  KDL::Rotation test_orientation {1, 0, 0, 0, -1, 0, 0, 0, -1};

  Json::Value data;

  long long ts;
  std::string pkg_path;
  std::string dir_path;
  std::string remove_dir_path;
  std::ofstream myfile;
  boost::filesystem::path dir;
  boost::filesystem::path remove_dir;
  pkg_path = ros::package::getPath("trajectory_generator");
  // std::cout << "pkg_path: " << pkg_path << std::endl;
  remove_dir_path = pkg_path + "/generated_trajectories/cpp/latest_batch";
  remove_dir = boost::filesystem::path(remove_dir_path);
  
  while (generated_trajectories < batch_count)
  {
    exception_count = 0;
    joint_trajectory.clear();
    eef_trajectory.clear();
    fk_eef_trajectory.clear();
    do
    {
      try
      {
        path_error = false;
        noisy_waypoints = starting_waypoints;
        path = new KDL::Path_RoundedComposite(RADIUS, EQRADIUS, orient, true);
        for (unsigned i = 0; i < noisy_waypoints.size(); i++)
        {
          if (i != 0) //  we do not add noise to the starting waypoint
          {
            for (unsigned ii = 0; ii < noisy_waypoints[i].size(); ii++)
            {
              max_noise = NOISE_MATRIX[ii]["noise_mean"] + 3*NOISE_MATRIX[ii]["noise_stddev"];
              min_noise = NOISE_MATRIX[ii]["noise_mean"] - 3*NOISE_MATRIX[ii]["noise_stddev"];
              if (i == 1 && ii == 2)
              {
                max_noise = abs(noisy_waypoints[0][2] - starting_waypoints[1][2]);
              }
              if (i == 2 && ii == 2)
              {
                min_noise = -abs(noisy_waypoints[1][2] - starting_waypoints[2][2]);
              }
              distribution = std::normal_distribution<double> (NOISE_MATRIX[ii]["noise_mean"], NOISE_MATRIX[ii]["noise_stddev"]);
              noise_generation_counter = 0;
              do
              {
                if (noise_generation_counter >= MAX_NOISE_GENERATION_ATTEMPTS)
                {
                  std::cout << "PROGRAM ABORTED: 'Couldn't generate noise inside the interval: (" << min_noise / 100 << ", " << max_noise / 100 << ") after " << MAX_NOISE_GENERATION_ATTEMPTS << " attempts'" << std::endl;
                  return 0;
                }
                // std::cout << "Generating noise: " << noise_generation_counter << std::endl;
                noise = distribution(generator) / 100.0;
                noise_generation_counter++;
              }
              while (!(noise < max_noise / 100.0 && noise > min_noise / 100.0));
              noisy_waypoints[i][ii] += noise;
              if (i == 1 && ii == 0)  //  x coordinate of the second waypoint -> noisy_release_x_coordinate
              {
                noisy_release_x_coordinate = noisy_waypoints[i][ii];
              }
              // std::cout << "\nnoise: " << noise << std::endl;
              // std::cout << "noisy_waypoints[" << i << "][" << ii << "]: " << noisy_waypoints[i][ii] << std::endl;
            }
          }
          path -> Add(KDL::Frame(KDL::Vector(noisy_waypoints[i][0], noisy_waypoints[i][1], noisy_waypoints[i][2])));
        }
        path -> Finish();
      }
      catch(std::exception& e)
      {
        exception_count++;
        path_error = true;
        std::cerr << e.what() << '\n';
      }
      catch(...)
      {
        exception_count++;
        path_error = true;
        std::cout << "PATH GENERATION ERROR, DEFAULT EXCEPTION, generating a new path...\n";
      }
      if (exception_count >= MAX_PATH_GENERATION_ATTEMPTS)
      {
        std::cout << "PROGRAM ABORTED: 'Couldn't find a feasible path after " << MAX_PATH_GENERATION_ATTEMPTS << " attempts'" << std::endl;
        return 0;
      }
    }
    while (path_error);

    for (unsigned i = 0; i < nr_of_joints; i++)
    {
      last_joint_pos(i) = start_joint_pos_array[i];
    }

    ret = chainFkSolverPos.JntToCart(last_joint_pos, fk_current_eef_frame);
    // std::cout << "FK RET: " << ret << std::endl;
    starting_orientation = fk_current_eef_frame.M;

    path_length = path -> PathLength();
    ds = path_length / NUMBER_OF_SAMPLES;

    double treshold_length;
    // treshold_length = binary_search_treshold(*path, 0, path_length, noisy_release_x_coordinate, BINARY_SEARCH_TRESHOLD);
    treshold_length = minima_search(*path, ds, MAX_NUMBER_OF_INVERSIONS);

    // char c;
    // std::cin >> c;
    // return 0;
    
    std::vector<double> velocity_dist(NUMBER_OF_SAMPLES, 0);
    velocity_profile_generator(velocity_dist, NUMBER_OF_SAMPLES, RELEASE_FRAME, path_length, treshold_length);

    // double euler_X;
    // double euler_Y;
    // double euler_Z;
    // test_orientation.GetEulerZYX(euler_Z, euler_Y, euler_X);
    // std::cout << "test_orientation: " << test_orientation << std::endl;
    // std::cout << "euler_X: " << euler_X << std::endl;
    // std::cout << "euler_Y: " << euler_Y << std::endl;
    // std::cout << "euler_Z: " << euler_Z << std::endl;

    // starting_orientation.GetEulerZYX(euler_Z, euler_Y, euler_X);
    // std::cout << "starting_orientation: " << starting_orientation << std::endl;
    // std::cout << "euler_X: " << euler_X << std::endl;
    // std::cout << "euler_Y: " << euler_Y << std::endl;
    // std::cout << "euler_Z: " << euler_Z << std::endl;

    // // calculate the new joint coordinates for the starting point
    // current_eef_frame = path -> Pos(0);
    // current_eef_frame.M = test_orientation;
    // // current_eef_frame.M = starting_orientation;
    // ret = chainIkSolverPos.CartToJnt(last_joint_pos, current_eef_frame, last_joint_pos);
    // // std::cout << "New starting joint coordinates RET = " << ret << std::endl;

    // // for (unsigned i = 0; i < last_joint_pos.rows(); i++)
    // // {
    // //   std::cout << "last_joint_pos(" << i << "): " << last_joint_pos(i) << std::endl;
    // // }
    // joint_trajectory.push_back(last_joint_pos);
    // eef_trajectory.push_back(path -> Pos(0));

    // // ====================FK====================
    // ret = chainFkSolverPos.JntToCart(last_joint_pos, fk_current_eef_frame);
    // // std::cout << "FK RET: " << ret << std::endl;
    // fk_current_eef_pos = fk_current_eef_frame.p;
    // fk_eef_trajectory.push_back(fk_current_eef_frame);
    // // std::cout << "------------------------------\n";
    // // std::cout << "FK EEF pos: " << std::endl;
    // // std::cout << "\tx: " << fk_current_eef_pos.x() << std::endl;
    // // std::cout << "\ty: " << fk_current_eef_pos.y() << std::endl;
    // // std::cout << "\tz: " << fk_current_eef_pos.z() << std::endl;
    // // std::cout << "------------------------------\n";
    // // ====================END FK====================

    current_s = 0;
    for (unsigned i = 0; i < NUMBER_OF_SAMPLES; i++)
    {
      // current_s += ds;
      current_s += velocity_dist[i];
      current_eef_frame = path -> Pos(current_s);
      current_eef_frame.M = test_orientation;
      // current_eef_frame.M = starting_orientation;
      eef_trajectory.push_back(current_eef_frame);

      // current_eef_frame.M.GetEulerZYX(Z, Y, X);
      // std::cout << "EEF frame orientation" << std::endl;
      // std::cout << "\t\t\tX: " << X << std::endl;
      // std::cout << "\t\t\tY: " << Y << std::endl;
      // std::cout << "\t\t\tZ: " << Z << std::endl;

      ret = chainIkSolverPos.CartToJnt(last_joint_pos, current_eef_frame, q_out);
      // std::cout << "RET TRUE: " << ret << std::endl;
      joint_trajectory.push_back(q_out);

      ret = chainFkSolverPos.JntToCart(q_out, fk_current_eef_frame);
      // std::cout << "------------------------------\n";
      // std::cout << "FK RET: " << ret << std::endl;
      // std::cout << "FK EEF pos: " << std::endl;
      // std::cout << "\tx: " << fk_current_eef_frame.p.x() << std::endl;
      // std::cout << "\ty: " << fk_current_eef_frame.p.y() << std::endl;
      // std::cout << "\tz: " << fk_current_eef_frame.p.z() << std::endl;
      // std::cout << "------------------------------\n";
      fk_eef_trajectory.push_back(fk_current_eef_frame);

      // std::cout << "------------------------------\n";
      // std::cout << "Point " << i << ": " << std::endl;
      // std::cout << "x: " << current_eef_frame.p.x() << std::endl;
      // std::cout << "y: " << current_eef_frame.p.y() << std::endl;
      // std::cout << "z: " << current_eef_frame.p.z() << std::endl;
      // std::cout << "q_out:\n" << q_out.data << std::endl;
      // std::cout << "------------------------------\n";
      last_joint_pos = q_out;
    }

    // ====================JSON====================
    data = Json::Value();
    data["realease_frame"] = RELEASE_FRAME;
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

    ts = std::chrono::system_clock::now().time_since_epoch().count();
    // std::cout << "time: " << ts << std::endl;
    if (is_batch)
    {
      if (generated_trajectories == 0)
      {
        if(boost::filesystem::exists(remove_dir))
        {
          boost::filesystem::remove_all(remove_dir);
        }
        dir_path = pkg_path + "/generated_trajectories/cpp/latest_batch";
        dir = boost::filesystem::path(dir_path);
        if(!(boost::filesystem::exists(dir)))
        {
          if (boost::filesystem::create_directories(dir))
          {
            std::cout << "....'latest_batch' folder Successfully Created!" << std::endl;
          }
          else
          {
            std::cout << "....ERROR 'latest_batch' folder Couldn't Be Created!" << std::endl;
          }
        }
      }
      myfile.open(dir_path + "/" + std::to_string(generated_trajectories) + ".txt");
      if (myfile.is_open())
      {
        myfile << data << std::endl;
        myfile.close();
        std::cout << "Trajectories generated: " << generated_trajectories + 1 << std::endl;
      }
    }
    else
    {
      dir_path = pkg_path + "/generated_trajectories/cpp/latest";
      dir = boost::filesystem::path(dir_path);
      if(!(boost::filesystem::exists(dir)))
      {
        if (boost::filesystem::create_directories(dir))
        {
          std::cout << "....'latest' folder Successfully Created!" << std::endl;
        }
        else
        {
          std::cout << "....ERROR 'latest' folder Couldn't Be Created!" << std::endl;
        }
      }
      myfile.open(dir_path + "/trajectories.txt");
      if (myfile.is_open())
      {
        myfile << data << std::endl;
        myfile.close();
      }
    }

    // dir_path = pkg_path + "/generated_trajectories/cpp/" + std::to_string(ts);
    // dir = boost::filesystem::path(dir_path);
    // if(!(boost::filesystem::exists(dir)))
    // {
    //   if (boost::filesystem::create_directories(dir))
    //   {
    //     std::cout << "....'timestamp' folder Successfully Created!" << std::endl;
    //   }
    //   else
    //   {
    //     std::cout << "....ERROR 'timestamp' folder Couldn't Be Created!" << std::endl;
    //   }
    // }
    // myfile.open(dir_path + "/trajectories.txt");
    // if (myfile.is_open())
    // {
    //   myfile << data << std::endl;
    //   myfile.close();
    // }
    // ====================END JSON====================
    generated_trajectories++;
  }
  // int n = path -> GetNrOfSegments();
  // double length = path -> GetLengthToEndOfSegment(n);
  // std::cout << "my_tree.getNrOfJoints(): " << my_tree.getNrOfJoints() << std::endl;
  // std::cout << "path -> GetNrOfSegments(): " << path -> GetNrOfSegments() << std::endl;
  // std::cout << "path -> GetLengthToEndOfSegment(" << path -> GetNrOfSegments()-1 << "): " << path -> GetLengthToEndOfSegment(n-1) << std::endl;
  // std::cout << "path -> PathLength(): " << path -> PathLength() << std::endl;
  // std::cout << "exception_count: " << exception_count << std::endl;

  return 0;
}