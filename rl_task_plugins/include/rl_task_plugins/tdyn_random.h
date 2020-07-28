// The HiQP Control Framework, an optimal control framework targeted at robotics
// Copyright (C) 2016 Marcus A Johansson
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef HIQP_TDYN_RANDOM_H
#define HIQP_TDYN_RANDOM_H

#include <hiqp/robot_state.h>
#include <hiqp/task_dynamics.h>
#include <ros/ros.h>
#include "std_msgs/String.h"
#include <pluginlib/class_loader.h>
#include <kdl/treefksolverpos_recursive.hpp>
#include <kdl/treejnttojacsolver.hpp>
#include <kdl/chainfksolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/frames.hpp>
#include <kdl/frames_io.hpp>
#include <iostream>
#include <stdio.h>
#include <chrono>

namespace hiqp
{
  namespace tasks
  {

    struct KinematicQuantities {
      std::string frame_id_;
      KDL::Jacobian ee_J_;
      KDL::Frame ee_frame_;
      KDL::Vector ee_p_;
    };

  /*! \brief Random task error behavior? 
   *  \author Jens Lundell, Todor Stoyanov */  
    class TDynRandom : public TaskDynamics {
    public:

      inline TDynRandom() : TaskDynamics() {}
      
      TDynRandom(std::shared_ptr<GeometricPrimitiveMap> geom_prim_map,
       std::shared_ptr<Visualizer> visualizer);

      ~TDynRandom() noexcept {}

      int init(const std::vector<std::string>& parameters, 
              RobotStatePtr robot_state, 
              const Eigen::VectorXd& e_initial, 
              const Eigen::VectorXd& e_dot_initial, 
              const Eigen::VectorXd& e_final, 
              const Eigen::VectorXd& e_dot_final);

      int update(const RobotStatePtr robot_state,
                 const TaskDefinitionPtr def);

      int monitor();

      int pointForwardKinematics(
        std::vector<KinematicQuantities>& kin_q_list,
        const std::shared_ptr<geometric_primitives::GeometricPoint>& point,
        RobotStatePtr const robot_state) const;

  /*! Helper function which computes ee pose and Jacobian w.r.t. a given frame*/
      int forwardKinematics(KinematicQuantities& kin_q,
        RobotStatePtr const robot_state) const;

    private:
      TDynRandom(const TDynRandom& other) = delete;
      TDynRandom(TDynRandom&& other) = delete;
      TDynRandom& operator=(const TDynRandom& other) = delete;
      TDynRandom& operator=(TDynRandom&& other) noexcept = delete;

      std::default_random_engine generator;
      std::normal_distribution<double> dist;
      ros::Publisher starting_pub_;
      std_msgs::String msg_;
      std::shared_ptr<KDL::TreeFkSolverPos_recursive> fk_solver_pos_;
      std::shared_ptr<KDL::TreeJntToJacSolver> fk_solver_jac_;

      ros::NodeHandle nh_;

    };

} // namespace tasks

} // namespace hiqp

#endif // include guard
