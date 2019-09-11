#include "ros/ros.h"
#include "trajectory_generator/trajectory_safety_check.h"

#include <cmath>
#include <functional>
#include <thread>
#include <mutex>
#include <string>
#include <vector>

#include <ros/init.h>
#include <ros/node_handle.h>
#include <ros/rate.h>
#include <ros/spinner.h>
#include <sensor_msgs/JointState.h>

#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chain.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <boost/program_options.hpp>

bool add(trajectory_generator::trajectory_safety_check::Request &req,
         trajectory_generator::trajectory_safety_check::Response &res)
{
    res.sum = req.a + req.b;
    ROS_INFO("request: x=%ld, y=%ld", (long int)req.a, (long int)req.b);
    ROS_INFO("sending back response: [%ld]", (long int)res.sum);
    return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "trajectory_safety_check_server");
    ros::NodeHandle node_handle;
    std::string robot_ip;

    bool is_simulation = false;
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("sim,s", po::bool_switch()->default_value(false), "simulation flag")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 1;
    }

    if (vm.count("sim"))
    {
        is_simulation = vm["sim"].as<bool>();
    }

    KDL::Tree my_tree{};
    ros::NodeHandle node;
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

    TiXmlDocument tiny_doc;
    tiny_doc.Parse(robot_desc_string.c_str());
    std::map<std::string, std::map<std::string, double>> joint_limits;

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
        }
    }

    KDL::Chain my_chain = KDL::Chain{};
    std::vector<std::string> chain_segments_names;
    my_tree.getChain("world", "panda_hand", my_chain);
    unsigned nr_of_joints = my_chain.getNrOfJoints();
    KDL::JntArray joints_pos{nr_of_joints};
    KDL::Frame eef_frame;

    KDL::ChainFkSolverPos_recursive chainFkSolverPos{my_chain};

    if (!node_handle.getParam("/panda/franka_control/robot_ip", robot_ip))
    {
        ROS_ERROR("trajectory_safety_check_node: Could not parse robot_ip parameter");
        return -1;
    }

    auto safety_check_handler = [&my_chain, &chainFkSolverPos, &joints_pos, &eef_frame, nr_of_joints](trajectory_generator::trajectory_safety_check::Request& req, trajectory_generator::trajectory_safety_check::Response& res)
    {
        int ret = 0;
        for (int i = 0; i < req.joints_pos.size(); i++) {
            joints_pos(i % 7) = req.joints_pos[i];
            if (i % 7 == 6) {
                ret = chainFkSolverPos.JntToCart(joints_pos, eef_frame);
                std::cout << "\n\ti = " << i << std::endl;
                std::cout << "eef_frame.p.x() = " << eef_frame.p.x() << std::endl;
                std::cout << "eef_frame.p.y() = " << eef_frame.p.y() << std::endl;
                std::cout << "eef_frame.p.z() = " << eef_frame.p.z() << std::endl;
            }
        }
        res.sum = nr_of_joints;
        return "test";
    };

    ros::ServiceServer service = node_handle.advertiseService<trajectory_generator::trajectory_safety_check::Request, trajectory_generator::trajectory_safety_check::Response>("trajectory_safety_check", safety_check_handler);
    // ros::ServiceServer service = node_handle.advertiseService("trajectory_safety_check", add);
    
    ROS_INFO("Ready to add two ints.");
    ros::spin();

    return 0;
}