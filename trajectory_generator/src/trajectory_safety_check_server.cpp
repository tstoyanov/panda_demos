#include "ros/ros.h"
#include "trajectory_generator/trajectory_safety_check.h"
#include "trajectory_generator/trajectory_safety_check_z_setter.h"
#include "trajectory_generator/trajectory_safety_check_z_getter.h"
#include "trajectory_generator/trajectory_safety_check_y_setter.h"
#include "trajectory_generator/trajectory_safety_check_y_getter.h"

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

#include <math.h>

#define PI 3.14159265

int main(int argc, char **argv)
{
    std::string node_name = "trajectory_safety_check_node";
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
        ROS_ERROR("%s: Failed to construct kdl tree", node_name.c_str());
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
        joint_limits[joint_name].insert({{"velocity", std::stod(tiny_joint->FirstChild("limit")->ToElement()->Attribute("velocity"))}});
        }
    }

    KDL::Chain my_chain = KDL::Chain{};
    my_tree.getChain("world", "panda_hand", my_chain);
    unsigned nr_of_joints = my_chain.getNrOfJoints();
    KDL::JntArray joints_pos{nr_of_joints};
    KDL::JntArray last_joints_pos{nr_of_joints};
    KDL::Frame eef_frame;

    KDL::ChainFkSolverPos_recursive chainFkSolverPos{my_chain};

    // if (!node_handle.getParam("/panda/franka_control/robot_ip", robot_ip))
    // {
    //     ROS_ERROR("%s: Could not parse robot_ip parameter", node_name.c_str());
    //     return -1;
    // }

    double z_lower_limit;
    if (!node_handle.getParam("/trajectory_safety_check_server/z_lower_limit", z_lower_limit) && !node_handle.getParam("/panda/trajectory_safety_check_server/z_lower_limit", z_lower_limit))
    {
        z_lower_limit = 0.85671;
        ROS_ERROR("%s: Could not parse 'z_lower_limit' parameter. Using default value '0.85671' instead", node_name.c_str());
    }
    else
    {
        ROS_INFO("%s: Found 'z_lower_limit' parameter with value: %f", node_name.c_str(), z_lower_limit);
    }

    double z_upper_limit;
    if (!node_handle.getParam("/trajectory_safety_check_server/z_upper_limit", z_upper_limit) && !node_handle.getParam("/panda/trajectory_safety_check_server/z_upper_limit", z_upper_limit))
    {
        z_upper_limit = 0.86671;
        ROS_ERROR("%s: Could not parse 'z_upper_limit' parameter. Using default value '0.86671' instead", node_name.c_str());
    }
    else
    {
        ROS_INFO("%s: Found 'z_upper_limit' parameter with value: %f", node_name.c_str(), z_upper_limit);
    }

    double y_lower_limit;
    if (!node_handle.getParam("/trajectory_safety_check_server/y_lower_limit", y_lower_limit) && !node_handle.getParam("/panda/trajectory_safety_check_server/y_lower_limit", y_lower_limit))
    {
        y_lower_limit = -0.28;
        ROS_ERROR("%s: Could not parse 'y_lower_limit' parameter. Using default value '-0.16' instead", node_name.c_str());
    }
    else
    {
        ROS_INFO("%s: Found 'y_lower_limit' parameter with value: %f", node_name.c_str(), y_lower_limit);
    }

    double y_upper_limit;
    if (!node_handle.getParam("/trajectory_safety_check_server/y_upper_limit", y_upper_limit) && !node_handle.getParam("/panda/trajectory_safety_check_server/y_upper_limit", y_upper_limit))
    {
        y_upper_limit = -0.12;
        ROS_ERROR("%s: Could not parse 'y_upper_limit' parameter. Using default value '0.0' instead", node_name.c_str());
    }
    else
    {
        ROS_INFO("%s: Found 'y_upper_limit' parameter with value: %f", node_name.c_str(), y_upper_limit);
    }
    

    auto safety_check_handler = [&my_chain, &chainFkSolverPos, &joint_names, &joint_limits, &joints_pos, &last_joints_pos, &eef_frame, &z_lower_limit, &z_upper_limit, &y_lower_limit, &y_upper_limit, nr_of_joints](trajectory_generator::trajectory_safety_check::Request& req, trajectory_generator::trajectory_safety_check::Response& res)
    {
        int ret = 0;
        int joint_index;
        std::string joint_name;
        double rel_angle;
        double initial_x;
        double initial_y;
        double initial_z;
        double final_x;
        double final_y;
        double final_z;
        double x;
        double y;
        double z;
        double current_vel;
        double min_execution_time;
        int unsafe_pt = 0;
        double avg_distance = 0;
        std::vector<double> fk_y;
        std::vector<double> fk_z;
        std::vector<double> joints_min_execution_time;
        res.is_safe = true;
        res.too_left = false;
        res.too_right = false;
        res.too_high = false;
        res.too_low = false;
        res.too_fast = false;
        res.error = false;
        res.unsafe_pts = 0;
        res.z_unsafe_pts = 0;
        res.y_unsafe_pts = 0;
        for (int i = 0; i < req.joints_pos.size(); i++) {
            unsafe_pt = 0;
            joint_index = i % nr_of_joints;
            joint_name = joint_names[joint_index];
            joints_pos(joint_index) = req.joints_pos[i];
            if (i < 7)
            {
                res.joints_min_execution_time.push_back(-1);
                res.joints_too_fast.push_back(0);
                // min_exe_time_it = joints_min_execution_time.find(joint_name); 
                // if (min_exe_time_it != joints_min_execution_time.end())
                // {
                //     min_exe_time_it->second = -1;
                // }
                // else
                // {
                //     joints_min_execution_time.insert({joint_name, -1});
                // }
            }
            else
            {
                current_vel = joints_pos(joint_index) - last_joints_pos(joint_index);
                min_execution_time = fabs(current_vel) / joint_limits[joint_name]["velocity"] * req.joints_pos.size();
                if (req.execution_time > 0)
                {
                    if (req.execution_time < (min_execution_time*1000000000))
                    {
                        // unsafe_pt = 1;
                        res.joints_too_fast[joint_index]++;
                        res.too_fast = true;
                    }
                }
                if (res.joints_min_execution_time[joint_index] < min_execution_time)
                {
                    res.joints_min_execution_time[joint_index] = min_execution_time;
                }
                // min_exe_time_it = joints_min_execution_time.find(joint_name); 
                // if (min_exe_time_it != joints_min_execution_time.end() && min_exe_time_it->second < min_execution_time)
                // {
                //     min_exe_time_it->second = min_execution_time;
                // }
            }
            last_joints_pos(joint_index) = joints_pos(joint_index);

            if (i % 7 == 6) {
                ret = chainFkSolverPos.JntToCart(joints_pos, eef_frame);
                if (i == 6)
                {
                    initial_x = eef_frame.p.x();
                    initial_y = eef_frame.p.y();
                    initial_z = eef_frame.p.z();
                }
                fk_y.push_back(eef_frame.p.y());
                fk_z.push_back(eef_frame.p.z());
                avg_distance += std::abs(eef_frame.p.z() - z_lower_limit);
                if (eef_frame.p.y() <= y_lower_limit)
                {
                    unsafe_pt = 1;
                    res.y_unsafe_pts++;
                    res.too_right = true;
                }
                if (eef_frame.p.y() >= y_upper_limit)
                {
                    unsafe_pt = 1;
                    res.y_unsafe_pts++;
                    res.too_left = true;
                }
                if (eef_frame.p.z() < z_lower_limit)
                {
                    unsafe_pt = 1;
                    res.z_unsafe_pts++;
                    res.too_low = true;
                }
                if ((i+1) / nr_of_joints < 95 && eef_frame.p.z() > z_upper_limit)
                {
                    unsafe_pt = 1;
                    res.z_unsafe_pts++;
                    res.too_high = true;
                }
                res.unsafe_pts += unsafe_pt;
                if ((i+1) / nr_of_joints == 94)
                {
                    final_x = eef_frame.p.x();
                    final_y = eef_frame.p.y();
                    final_z = eef_frame.p.z();
                }
                if ((i+1) / nr_of_joints == 95) // release frame
                {
                    final_x += eef_frame.p.x();
                    final_y += eef_frame.p.y();
                    final_z += eef_frame.p.z();
                }
                if ((i+1) / nr_of_joints == 96)
                {
                    final_x += eef_frame.p.x();
                    final_y += eef_frame.p.y();
                    final_z += eef_frame.p.z();
                    
                    final_x /= 3;
                    final_y /= 3;
                    final_z /= 3;

                    x = final_x - initial_x;
                    y = initial_y - final_y; // inverted because the y axis is in the opposite direction
                    res.rel_angle = atan2(x, y) * 180 / PI; // x and y are inverted because the axes in the cartesian space are inverted
                }
            }
        }
        if (res.unsafe_pts != 0) {
            res.is_safe = false;
        }
        res.fk_y = fk_y;
        res.fk_z = fk_z;
        res.avg_distance = avg_distance / req.joints_pos.size();
        return 1;
    };

    auto z_setter_handler = [&z_lower_limit, &z_upper_limit, &node_name](trajectory_generator::trajectory_safety_check_z_setter::Request& req, trajectory_generator::trajectory_safety_check_z_setter::Response& res)
    {
        if (req.update_lower) {
            z_lower_limit = req.z_lower_limit;
            ROS_INFO("%s: Value of 'z_lower_limit' succesfully updated to: %f", node_name.c_str(), z_lower_limit);
        }
        if (req.update_upper) {
            z_upper_limit = req.z_upper_limit;
            ROS_INFO("%s: Value of 'z_upper_limit' succesfully updated to: %f", node_name.c_str(), z_upper_limit);
        }
        res.updated_lower = z_lower_limit;
        res.updated_upper = z_upper_limit;
        res.error = false;
        return 1;
    };

    auto z_getter_handler = [&z_lower_limit, &z_upper_limit](trajectory_generator::trajectory_safety_check_z_getter::Request& req, trajectory_generator::trajectory_safety_check_z_getter::Response& res){
        res.z_lower = z_lower_limit;
        res.z_upper = z_upper_limit;
        return 1;
    };

    auto y_setter_handler = [&y_lower_limit, &y_upper_limit, &node_name](trajectory_generator::trajectory_safety_check_y_setter::Request& req, trajectory_generator::trajectory_safety_check_y_setter::Response& res)
    {
        if (req.update_lower) {
            y_lower_limit = req.y_lower_limit;
            ROS_INFO("%s: Value of 'y_lower_limit' succesfully updated to: %f", node_name.c_str(), y_lower_limit);
        }
        if (req.update_upper) {
            y_upper_limit = req.y_upper_limit;
            ROS_INFO("%s: Value of 'y_upper_limit' succesfully updated to: %f", node_name.c_str(), y_upper_limit);
        }
        res.updated_lower = y_lower_limit;
        res.updated_upper = y_upper_limit;
        res.error = false;
        return 1;
    };

    auto y_getter_handler = [&y_lower_limit, &y_upper_limit](trajectory_generator::trajectory_safety_check_y_getter::Request& req, trajectory_generator::trajectory_safety_check_y_getter::Response& res){
        res.y_lower = y_lower_limit;
        res.y_upper = y_upper_limit;
        return 1;
    };

    ros::ServiceServer safety_check_service = node_handle.advertiseService<trajectory_generator::trajectory_safety_check::Request, trajectory_generator::trajectory_safety_check::Response>("trajectory_safety_check", safety_check_handler);
    ros::ServiceServer z_setter_service = node_handle.advertiseService<trajectory_generator::trajectory_safety_check_z_setter::Request, trajectory_generator::trajectory_safety_check_z_setter::Response>("trajectory_safety_check_z_setter", z_setter_handler);
    ros::ServiceServer z_getter_service = node_handle.advertiseService<trajectory_generator::trajectory_safety_check_z_getter::Request, trajectory_generator::trajectory_safety_check_z_getter::Response>("trajectory_safety_check_z_getter", z_getter_handler);
    ros::ServiceServer y_setter_service = node_handle.advertiseService<trajectory_generator::trajectory_safety_check_y_setter::Request, trajectory_generator::trajectory_safety_check_y_setter::Response>("trajectory_safety_check_y_setter", y_setter_handler);
    ros::ServiceServer y_getter_service = node_handle.advertiseService<trajectory_generator::trajectory_safety_check_y_getter::Request, trajectory_generator::trajectory_safety_check_y_getter::Response>("trajectory_safety_check_y_getter", y_getter_handler);
    ros::spin();

    return 0;
}