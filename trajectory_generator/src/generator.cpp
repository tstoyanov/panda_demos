#include <ros/ros.h>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/trajectory_segment.hpp>
#include <kdl/path_roundedcomposite.hpp>
#include <kdl/velocityprofile_spline.hpp>
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

#define PI 3.14159265

#define RADIUS 1
#define EQRADIUS 1

#define MAX_PATH_GENERATION_ATTEMPTS 100
#define MAX_NOISE_GENERATION_ATTEMPTS 100

#define DECELERATION_OFFSET 5
#define DECELERATION_TIME 0.25 // duration of the deceleration in seconds
#define DECELERATION_FRAMES 5  // number of frames during deceleration

#define NUMBER_OF_SAMPLES 100
#define RELEASE_FRAME NUMBER_OF_SAMPLES - DECELERATION_OFFSET
// #define RELEASE_FRAME 0.8 * NUMBER_OF_SAMPLES

#define BINARY_SEARCH_TRESHOLD 0.0001 // meters
#define MAX_NUMBER_OF_INVERSIONS 10

// [68%, 95%, 99.7%] of the noise falls inside MEAN +/- [1, 2, 3]*stddev (1 = +/- 3m? => 3cm in the end)
std::vector<std::map<std::string, double>> NOISE_MATRIX = { // 1 will be converted to 1 cm for now
    // good
    {
      // x
      {"mean", 0},
      {"stddev", 1} // 3
    },
    {
      // y
      {"mean", 0},
      {"stddev", 10}},
    //   {"stddev", 0.3}},
    {
      // z
      {"mean", 0},
      {"stddev", 1} // 2
    }};

// std::vector<double> distance_from_release = {

// }

double binary_search_treshold(KDL::Path_RoundedComposite &path, double lower_bound, double upper_bound, const double &x_value, const double &treshold)
{
  double mid = (upper_bound + lower_bound) / 2;
  double mid_x_value = path.Pos(mid).p.x();
  // if (mid_x_value <= (x_value + treshold) && mid_x_value >= (x_value - treshold))
  if (fabs(mid_x_value - x_value) <= treshold)
  {
    return mid;
  }
  else
  {
    if (mid_x_value < x_value)
    {
      return binary_search_treshold(path, mid, upper_bound, x_value, treshold);
    }
    else
    {
      return binary_search_treshold(path, lower_bound, mid, x_value, treshold);
    }
  }
}

double minima_search(KDL::Path_RoundedComposite &path, double ds, int max_number_of_inversions)
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
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  return std::string(result, (count > 0) ? count : 0);
}

int main(int argc, char **argv)
{
  bool is_new = false;
  bool is_batch = false;
  bool is_simulation = false;
  bool no_noise = false;
  int batch_count = 1;
  int generated_trajectories = 0;
  double release_velocity = 1;
  double trajectory_duration = 1.45;    // seconds
//   double trajectory_duration = 1.45;    // seconds
  double dt = trajectory_duration / NUMBER_OF_SAMPLES;
  namespace po = boost::program_options;
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("batch,b", po::value<int>(&batch_count), "set the number of trajectories to generate")
    ("sim,s", po::bool_switch()->default_value(false), "simulation flag")
    ("no-noise,n", po::bool_switch()->default_value(false), "no-noise flag")
    ("new", po::bool_switch()->default_value(false), "new trajectory generation flag")
    ("duration,t", po::value<double>(&trajectory_duration), "duration of the trajectory in seconds")
    ("velocity,v", po::value<double>(&release_velocity), "release velocity in [m/s]")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
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
  }

  if (vm.count("no-noise"))
  {
    no_noise = vm["no-noise"].as<bool>();
  }

  if (vm.count("new"))
  {
    is_new = vm["new"].as<bool>();
  }

  if (vm.count("duration"))
  {
    trajectory_duration = vm["duration"].as<double>();
    dt = trajectory_duration / NUMBER_OF_SAMPLES;
  }

  if (vm.count("velocity"))
  {
    release_velocity = vm["velocity"].as<double>();
  }

  // if (argc > 1)
  // {
  //   RADIUS = std::stod(argv[1]);
  //   EQRADIUS = std::stod(argv[1]);
  // }
  ros::init(argc, argv, "myGenerator");
  KDL::Tree my_tree{};
  ros::NodeHandle node;
  ros::Rate loop_rate(2);

  // ===== getting URDF model and joint names form param server =====
  std::string robot_desc_string;
  std::vector<std::string> joint_names;
  if (is_simulation)
  {
    std::cout << "SIMULATION MODE" << std::endl;
    node.param("robot_description", robot_desc_string, std::string());
    node.getParam("position_joint_trajectory_controller/joints", joint_names);
  }
  else
  {
    node.param("/panda/robot_description", robot_desc_string, std::string());
    node.getParam("/panda/position_joint_trajectory_controller/joints", joint_names);
  }

  if (!kdl_parser::treeFromString(robot_desc_string, my_tree))
  {
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
  std::map<std::string, double> joints_min_execution_time;

  TiXmlHandle doc_handle{&tiny_doc};
  std::string joint_name;
  std::vector<std::string>::iterator it;
  for (TiXmlElement *tiny_joint = doc_handle.FirstChild("robot").Child("joint", 0).ToElement(); tiny_joint; tiny_joint = tiny_joint->NextSiblingElement("joint"))
  {
    joint_name = tiny_joint->Attribute("name");
    it = std::find(joint_names.begin(), joint_names.end(), joint_name);
    if (it != joint_names.end())
    {
      joint_limits[joint_name].insert({{"lower", std::stod(tiny_joint->FirstChild("limit")->ToElement()->Attribute("lower"))}});
      joint_limits[joint_name].insert({{"upper", std::stod(tiny_joint->FirstChild("limit")->ToElement()->Attribute("upper"))}});
      joint_limits[joint_name].insert({{"velocity", std::stod(tiny_joint->FirstChild("limit")->ToElement()->Attribute("velocity"))}});
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

  KDL::Chain my_chain = KDL::Chain{};
  std::vector<std::string> chain_segments_names;
  my_tree.getChain("world", "panda_hand", my_chain);
  unsigned nr_of_joints = my_chain.getNrOfJoints();
  // std::cout << "my_chain.getNrOfJoints(): " << my_chain.getNrOfJoints() << std::endl;
  // for (unsigned i = 0; i < my_chain.getNrOfSegments(); i++)
  // {
  //   chain_segments_names.push_back(my_chain.getSegment(i).getName());
  //   std::cout << "\tchain segment" << i << ": " << my_chain.getSegment(i).getName() << std::endl;
  // }

  KDL::JntArray q_min{(unsigned)joint_names.size()};
  KDL::JntArray q_max{(unsigned)joint_names.size()};
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
  KDL::ChainFkSolverPos_recursive chainFkSolverPos{my_chain};
  KDL::ChainIkSolverVel_wdls chainIkSolverVel{my_chain};
  KDL::ChainIkSolverPos_NR_JL chainIkSolverPos{my_chain, q_min, q_max, chainFkSolverPos, chainIkSolverVel, max_iter, eps};

  double release_x_coordinate = 0.020281002938;
  double push_release_x_coordinate = 0.318281002938;
  double elbow_first_release_x_coordinate = -0.048281002938;
//   double release_x_coordinate = 0.088281002938;
  // double release_x_coordinate = 0.268281002938;
  double noisy_release_x_coordinate;
  // ========== WAYPOINTS ==========
  std::vector<std::vector<double>> starting_waypoints =
  {
    //   // elbow first
    //   // {x, y, z}
    //   {-0.431718997062, -0.2302002648095, 0.861710060669},
    //   {elbow_first_release_x_coordinate, -0.2302002648095, 0.861710060669},
    // //   {release_x_coordinate, -0.1102002648095, 0.861710060669},
    //   {0.058281002938, -0.2302002648095, 0.862710060669}
    
    //   // pushing
    //   // {x, y, z}
    //   {-0.181718997062, -0.1102002648095, 0.861710060669},
    //   {push_release_x_coordinate, -0.1102002648095, 0.861710060669},
    // //   {release_x_coordinate, -0.1102002648095, 0.861710060669},
    //   {0.408281002938, -0.1102002648095, 0.861710060669}
    
      // new urdf sliding height
      // {x, y, z}
      {-0.461718997062, -0.2302002648095, 0.861710060669},
      {release_x_coordinate, -0.2302002648095, 0.861710060669},
    //   {release_x_coordinate, -0.1102002648095, 0.861710060669},
    //   {0.118281002938, -0.1102002648095, 0.866710060669}
      {0.128281002938, -0.2302002648095, 0.866710060669}
      
    //   // sliding height
    //   // {x, y, z}
    //   {-0.481718997062, -0.1102002648095, 0.861710060669},
    //   {release_x_coordinate, -0.1102002648095, 0.861710060669},
    // //   {release_x_coordinate, -0.1102002648095, 0.861710060669},
    // //   {0.118281002938, -0.1102002648095, 0.866710060669}
    //   {0.108281002938, -0.1102002648095, 0.866710060669}


      // testing "safe" height
      // {x, y, z}
      // {-0.471718997062, -0.0112002648095, 0.931710060669},
      // {release_x_coordinate, -0.0112002648095, 0.931710060669},
      // {0.198281002938, -0.0112002648095, 0.936710060669}

      // // real test
      // // {x, y, z}
      // {-0.501718997062, -0.0112002648095, 0.856710060669},
      // {release_x_coordinate, -0.0112002648095, 0.856710060669},
      // {0.118281002938, -0.0112002648095, 0.861710060669}

      // good
      // {x, y, z}
      // {-0.401718997062, 0.0892002648095, 0.986710060669},
      // {release_x_coordinate, 0.0892002648095, 0.886710060669},
      // {0.568281002938, 0.0892002648095, 1.086710060669}

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

  KDL::RotationalInterpolation_SingleAxis *orient = new KDL::RotationalInterpolation_SingleAxis();
  KDL::Path_RoundedComposite *path;

  // ========== adding noise to the waypoints ==========
  bool path_error;
  double noise;
  int noise_generation_counter;
  std::vector<double> noisy_release_point;
  std::vector<std::vector<double>> noisy_waypoints;
  double max_noise;
  double min_noise;
  std::default_random_engine generator{std::random_device()()};
  std::normal_distribution<double> distribution;
  std::uniform_real_distribution<> real_distribution;
  int exception_count;

  double current_s;
  double current_t;
  double current_vel;
  double min_execution_time;
  KDL::Frame current_eef_frame;
  KDL::Frame fk_current_eef_frame;
  std::vector<KDL::Frame> eef_trajectory;
  std::vector<KDL::Frame> fk_eef_trajectory;
  KDL::Vector current_eef_pos;
  KDL::Vector fk_current_eef_pos;
  std::vector<KDL::JntArray> joint_trajectory;
  KDL::JntArray q_out{nr_of_joints};
  KDL::JntArray last_joint_pos{nr_of_joints};

  int ret;
  double path_length;
  double ds;

  double left_test[] = {-0.077235275171305, 1.1048619692529187, -0.8513244480763084, -1.4185140619442633, 0.9112366011341413, 2.131054658748485, -1.9308082009554988};
  double right_test[] = {-0.9117365299965204, 0.8700663910466688, 0.26306596901763923, -1.500111004846137, -0.19470499986410136, 2.4250543754889207, -1.4167937111554167};
  double bent_test[] = {-1.5673447221614194, 1.7283925611583961, 1.600207030158307, -1.4305450502590926, -1.8139610840135125, 1.6117727255345327, -0.879587150979955};
  double elbow_first[] = {0.10635736243363764, 1.4900843031226025, -1.3238531985717912, -1.3702379363758692, 1.4941375321812838, 1.818093645711391, -1.9917309573385749};
  
  double start_joint_pos_array[] = {-0.448125769162, 0.32964587676, -0.621680615641, -1.82515059054, 0.269715026327, 2.11438395741, -1.94274845254};
  double good_start_joint_pos_array[] = {-0.4385408868406217, 0.97796379816593138, -0.42510480666790013, -1.3566301885001291, 0.47105463493042737, 2.2513508344955504, -1.7757913197863211};
  double push_start_joint_pos_array[] = {-1.60864894845402, 1.6329093634789467, 1.4522765189189357, -2.085324071528737, -1.7060958243476456, 1.6604129666222465, -0.28491070685932524};
  double average_start_joint_pos_array[] = {0, 0, 0, -1.5708, 0, 1.8675, 0};
  double kdl_start_joint_pos_array[] = {-0.443379, 0.702188, -0.556869, -1.9368, -2.55769, 0.667764, -2.56121};

  double X;
  double Y;
  double Z;
  double W;

  double m;
  double c;
  double angle;
  double alpha;
  double beta;
  double gamma;
  double x_dist;
  double y_dist;

  KDL::Rotation starting_orientation;
  KDL::Rotation angle_orientation;
  KDL::Rotation test_orientation{1, 0, 0, 0, -1, 0, 0, 0, -1};

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

  std::vector<double> joints_release_ds;
  std::vector<double> joints_current_ds;
  std::vector<std::vector<double>> decelerating_frames;

  KDL::VelocityProfile_Spline *vel_prof = new KDL::VelocityProfile_Spline();
  KDL::Trajectory_Segment *trajectory;

  std::map<std::string, double> m_map;
  std::map<std::string, double> c_map;
  std::map<std::string, double>::iterator mc_it;
  std::map<std::string, double>::iterator min_exe_time_it;

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
        noisy_release_point = starting_waypoints[1];
        // noisy_waypoints = starting_waypoints;
        path = new KDL::Path_RoundedComposite(RADIUS, EQRADIUS, orient, true);

        // ========== old noise generation ==========
        // for (unsigned i = 0; i < noisy_waypoints.size(); i++)
        // {
        //   if (i != 0 && i != 2) //  we do not add noise to the starting waypoint (and ending)
        //   {
        //     for (unsigned ii = 0; ii < noisy_waypoints[i].size(); ii++)
        //     {
        //       max_noise = NOISE_MATRIX[ii]["mean"] + 3*NOISE_MATRIX[ii]["stddev"];
        //       min_noise = NOISE_MATRIX[ii]["mean"] - 3*NOISE_MATRIX[ii]["stddev"];
        //       if (i == 1 && ii == 2)
        //       {
        //         max_noise = abs(noisy_waypoints[0][2] - starting_waypoints[1][2]);
        //       }
        //       if (i == 2 && ii == 2)
        //       {
        //         min_noise = -abs(noisy_waypoints[1][2] - starting_waypoints[2][2]);
        //       }
        //       distribution = std::normal_distribution<double> (NOISE_MATRIX[ii]["mean"], NOISE_MATRIX[ii]["stddev"]);
        //       noise_generation_counter = 0;
        //       do
        //       {
        //         if (noise_generation_counter >= MAX_NOISE_GENERATION_ATTEMPTS)
        //         {
        //           std::cout << "PROGRAM ABORTED: 'Couldn't generate noise inside the interval: (" << min_noise / 100 << ", " << max_noise / 100 << ") after " << MAX_NOISE_GENERATION_ATTEMPTS << " attempts'" << std::endl;
        //           return 0;
        //         }
        //         // std::cout << "Generating noise: " << noise_generation_counter << std::endl;
        //         noise = distribution(generator) / 100.0;
        //         noise_generation_counter++;
        //       }
        //       while (!(noise < max_noise / 100.0 && noise > min_noise / 100.0));
        //       if (no_noise == false)
        //       {
        //         noisy_waypoints[i][ii] += noise;
        //       }
        //       if (i == 1 && ii == 0)  //  x coordinate of the second waypoint -> noisy_release_x_coordinate
        //       {
        //         noisy_release_x_coordinate = noisy_waypoints[i][ii];
        //       }
        //       // std::cout << "\nnoise: " << noise << std::endl;
        //       // std::cout << "noisy_waypoints[" << i << "][" << ii << "]: " << noisy_waypoints[i][ii] << std::endl;
        //     }
        //   }
        //   path -> Add(KDL::Frame(KDL::Vector(noisy_waypoints[i][0], noisy_waypoints[i][1], noisy_waypoints[i][2])));
        // }
        path->Add(KDL::Frame(KDL::Vector(starting_waypoints[0][0], starting_waypoints[0][1], starting_waypoints[0][2])));
        for (unsigned matrix_index = 0; matrix_index < NOISE_MATRIX.size(); matrix_index++)
        {
          real_distribution = std::uniform_real_distribution<double>(-NOISE_MATRIX[matrix_index]["stddev"], NOISE_MATRIX[matrix_index]["stddev"]);
          noise = real_distribution(generator) / 100.0;
        //   distribution = std::normal_distribution<double>(NOISE_MATRIX[matrix_index]["mean"], NOISE_MATRIX[matrix_index]["stddev"]);
        //   noise = distribution(generator) / 100.0;
          // if (matrix_index == 1)
          // {
          //   noisy_release_point[matrix_index] += 0.00;
          // }
          // if (matrix_index != 2)
          // {
          //   noisy_release_point[matrix_index] += noise;
          // }
          // else
          // { // we add only positive noise to the Z coordinate
          //   noisy_release_point[matrix_index] += abs(noise);
          // }
        }
        path->Add(KDL::Frame(KDL::Vector(noisy_release_point[0], noisy_release_point[1], noisy_release_point[2])));
        // path -> Add(KDL::Frame(KDL::Vector(noisy_release_point[0], noisy_release_point[1], noisy_release_point[2])));

        // calculating line equation
        m = (noisy_release_point[1] - starting_waypoints[0][1]) / (noisy_release_point[0] - starting_waypoints[0][0]);
        c = (noisy_release_point[1] - (m * noisy_release_point[0]));

        starting_waypoints[2][1] = m * starting_waypoints[2][0] + c;

        m = trunc(m*100)/100;
        c = trunc(c*100)/100;

        mc_it = m_map.find(std::to_string(m)); 
        if (mc_it != m_map.end()){
          mc_it->second = mc_it->second + 1;
        }
        else {
          m_map.insert({std::to_string(m), 1});
        }

        mc_it = c_map.find(std::to_string(c)); 
        if (mc_it != c_map.end()){
          mc_it->second = mc_it->second + 1;
        }
        else {
          c_map.insert({std::to_string(c), 1});
        }


        path->Add(KDL::Frame(KDL::Vector(starting_waypoints[2][0], starting_waypoints[2][1], starting_waypoints[2][2])));
        noisy_release_x_coordinate = noisy_release_point[0];
        path->Finish();
      }
      catch (std::exception &e)
      {
        exception_count++;
        path_error = true;
        std::cerr << e.what() << '\n';
      }
      catch (...)
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
    } while (path_error);

    for (unsigned i = 0; i < nr_of_joints; i++)
    {
    //   last_joint_pos(i) = push_start_joint_pos_array[i];
      last_joint_pos(i) = good_start_joint_pos_array[i];
      // last_joint_pos(i) = elbow_first[i];
    }

    ret = chainFkSolverPos.JntToCart(last_joint_pos, fk_current_eef_frame);
    // std::cout << "FK RET: " << ret << std::endl;
    starting_orientation = fk_current_eef_frame.M;



    x_dist = starting_waypoints[2][0] - starting_waypoints[0][0];
    y_dist = starting_waypoints[2][1] - starting_waypoints[0][1];
    // angle = fmod((atan2(y_dist, x_dist)*180/PI) + 360, 360);
    angle = fmod(atan2(y_dist, x_dist) + PI, 2*PI);
    // angle = angle*PI/180;
    // angle = atan2(y_dist, x_dist);
    starting_orientation.GetEulerZYX(alpha, gamma, beta);
    if (!is_batch)
    {
        std::cout << "alpha: " << alpha << std::endl;
        std::cout << "beta: " << beta << std::endl;
        std::cout << "gamma: " << gamma << std::endl;
        std::cout << "angle: " << angle << std::endl;
    }

    // starting_orientation = KDL::Rotation();
    // starting_orientation.DoRotZ(0);
    // starting_orientation.DoRotY(3);
    // starting_orientation.DoRotX(-0);

    // starting_orientation.GetEulerZYX(alpha, gamma, beta);
    // std::cout << "\nalpha: " << alpha << std::endl;
    // std::cout << "beta: " << beta << std::endl;
    // std::cout << "gamma: " << gamma << std::endl;

    path_length = path->PathLength();
    ds = path_length / NUMBER_OF_SAMPLES;
    double treshold_length;
    treshold_length = binary_search_treshold(*path, 0, path_length, noisy_release_x_coordinate, BINARY_SEARCH_TRESHOLD);
    // treshold_length = minima_search(*path, ds, MAX_NUMBER_OF_INVERSIONS);

    // char c;
    // std::cin >> c;
    // return 0;

    // =========================== OLD VELOCITY PROFILE ===========================
    std::vector<double> velocity_dist(NUMBER_OF_SAMPLES, 0);

    generate_velocity_profile(velocity_dist, NUMBER_OF_SAMPLES, RELEASE_FRAME, path_length, treshold_length);
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
    // ========================= END OLD VELOCITY PROFILE =========================

    // ======================== NEW TRAJECTORY ========================
    if (is_new)
    {
      release_velocity = 1.5;
      trajectory_duration = 1.45;
      // SetProfileDuration(double pos1, double vel1, double acc1, double pos2, double vel2, double acc2, double duration)
    //   vel_prof->SetProfileDuration(0.0, 0.0, 0.0, path_length, release_velocity, 0, trajectory_duration);
      // SetProfileDuration (double pos1, double vel1, double pos2, double vel2, double duration)
      vel_prof->SetProfileDuration(0.0, 0.0, path_length, release_velocity, trajectory_duration);
      // SetProfileDuration(double pos1, double pos2, double duration)
    //   vel_prof->SetProfileDuration(0.0, path_length, trajectory_duration);
      trajectory = new KDL::Trajectory_Segment(path, vel_prof);
      current_t = 0;
      for (unsigned i = 0; i <= NUMBER_OF_SAMPLES; i++)
      {
        current_eef_frame = trajectory->Pos(current_t);
        current_eef_frame.M = test_orientation;

        // current_eef_frame.M = starting_orientation;
        eef_trajectory.push_back(current_eef_frame);

        // current_eef_frame.M.GetEulerZYX(Z, Y, X);
        // std::cout << "EEF frame orientation" << std::endl;
        // std::cout << "\t\t\tX: " << X << std::endl;
        // std::cout << "\t\t\tY: " << Y << std::endl;
        // std::cout << "\t\t\tZ: " << Z << std::endl;

        ret = chainIkSolverPos.CartToJnt(last_joint_pos, current_eef_frame, q_out);
        for (unsigned joint_index = 0; joint_index < joint_names.size(); joint_index++)
        {
            joint_name = joint_names[joint_index];
            current_vel = q_out.data[joint_index] - last_joint_pos.data[joint_index];
            min_execution_time = fabs(current_vel) / joint_limits[joint_name]["velocity"] * NUMBER_OF_SAMPLES;
            if (i == 1)
            {
                joints_min_execution_time.insert({joint_name, min_execution_time});
            }
            else
            // else if (i != 1)
            {
                min_exe_time_it = joints_min_execution_time.find(joint_name); 
                if (min_exe_time_it != joints_min_execution_time.end() && min_exe_time_it->second < min_execution_time)
                {
                    min_exe_time_it->second = min_execution_time;
                }
            }
        }
        // current_eef_frame.p.x();
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
        current_t += dt;
      }
      // ====================== END NEW TRAJECTORY ======================
    }
    // // ======================== OLD TRAJECTORY ========================
    else
    {
      current_s = 0;
      for (unsigned i = 0; i < NUMBER_OF_SAMPLES; i++)
      {

        // current_s += ds;
        current_s += velocity_dist[i];
        current_eef_frame = path->Pos(current_s);
        angle_orientation = KDL::Rotation();
        angle_orientation.DoRotZ(angle);
        angle_orientation.DoRotY(3.14);
        angle_orientation.DoRotX(-0.02);
        current_eef_frame.M = angle_orientation;
        // current_eef_frame.M = test_orientation;
        // current_eef_frame.M = starting_orientation;
        eef_trajectory.push_back(current_eef_frame);

        // current_eef_frame.M.GetEulerZYX(Z, Y, X);
        // std::cout << "EEF frame orientation" << std::endl;
        // std::cout << "\t\t\tX: " << X << std::endl;
        // std::cout << "\t\t\tY: " << Y << std::endl;
        // std::cout << "\t\t\tZ: " << Z << std::endl;

        ret = chainIkSolverPos.CartToJnt(last_joint_pos, current_eef_frame, q_out);
        for (unsigned joint_index = 0; joint_index < joint_names.size(); joint_index++)
        {
            joint_name = joint_names[joint_index];
            current_vel = q_out.data[joint_index] - last_joint_pos.data[joint_index];
            min_execution_time = fabs(current_vel) / joint_limits[joint_name]["velocity"] * NUMBER_OF_SAMPLES;
            if (i == 1)
            {
                joints_min_execution_time.insert({joint_name, min_execution_time});
            }
            else
            // else if (i != 1)
            {
                min_exe_time_it = joints_min_execution_time.find(joint_name); 
                if (min_exe_time_it != joints_min_execution_time.end() && min_exe_time_it->second < min_execution_time)
                {
                    min_exe_time_it->second = min_execution_time;
                }
            }
        }
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
    }
    // // ====================== END OLD TRAJECTORY ======================

    // for (unsigned joint_index = 0; joint_index < joint_names.size(); joint_index++)
    // {
    //   joints_release_ds.push_back(joint_trajectory[joint_trajectory.size()].data[joint_index] - joint_trajectory[joint_trajectory.size()-1].data[joint_index]);
    // }

    // for (unsigned i = 0; i < DECELERATION_FRAMES; i++)
    // {
    //   for (unsigned joint_index = 0; joint_index < joint_names.size(); joint_index++)
    //   {
    //     joints_current_ds.push_back(joints_release_ds[joint_index] * (DECELERATION_FRAMES - i+1) / DECELERATION_FRAMES);
    //   }
    //   decelerating_frames.push_back(joints_current_ds);
    // }

    // ====================JSON====================
    data = Json::Value();
    data["realease_frame"] = RELEASE_FRAME;
    data["m"] = m;
    data["c"] = c;
    if (is_new)
    {
      data["trajectory_duration"] = trajectory_duration;
      data["release_velocity"] = release_velocity;
    }
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
        if (boost::filesystem::exists(remove_dir))
        {
          boost::filesystem::remove_all(remove_dir);
        }
        dir_path = pkg_path + "/generated_trajectories/cpp/latest_batch";
        dir = boost::filesystem::path(dir_path);
        if (!(boost::filesystem::exists(dir)))
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
        if (generated_trajectories % 100 == 99)
        {
          std::cout << "Trajectories generated: " << generated_trajectories + 1 << std::endl;
        }
      }
    }
    else
    {
      dir_path = pkg_path + "/generated_trajectories/cpp/latest";
      dir = boost::filesystem::path(dir_path);
      if (!(boost::filesystem::exists(dir)))
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
  std::cout << "m_map:" << std::endl;
  for(auto elem : m_map)
  {
    std::cout << elem.first << " - " << elem.second << std::endl;
  }
  std::cout << "c_map:" << std::endl;
  for(auto elem : c_map)
  {
    std::cout << elem.first << " - " << elem.second << std::endl;
  }
  std::cout << "joints_min_execution_time:" << std::endl;
  for(auto elem : joints_min_execution_time)
  {
    std::cout << elem.first << " - " << elem.second << std::endl;
  }
  return 0;
}