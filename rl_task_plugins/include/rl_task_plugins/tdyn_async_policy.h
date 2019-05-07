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

#ifndef HIQP_TDYN_ASYNC_POLICY_H
#define HIQP_TDYN_ASYNC_POLICY_H

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

#include <rl_task_plugins/DesiredErrorDynamicsMsg.h>

namespace hiqp
{
  namespace tasks
  {

  /*! \brief Random task error behavior? 
   *  \author Jens Lundell, Todor Stoyanov */  
    class TDynAsyncPolicy : public TaskDynamics {
    public:

      inline TDynAsyncPolicy() : TaskDynamics() {}
      
      TDynAsyncPolicy(std::shared_ptr<GeometricPrimitiveMap> geom_prim_map,
       std::shared_ptr<Visualizer> visualizer);

      ~TDynAsyncPolicy() noexcept {}

      int init(const std::vector<std::string>& parameters, 
              RobotStatePtr robot_state, 
              const Eigen::VectorXd& e_initial, 
              const Eigen::VectorXd& e_dot_initial, 
              const Eigen::VectorXd& e_final, 
              const Eigen::VectorXd& e_dot_final);

      int update(const RobotStatePtr robot_state, 
                const std::shared_ptr< TaskDefinition > def);

      int monitor();

    private:
      TDynAsyncPolicy(const TDynAsyncPolicy& other) = delete;
      TDynAsyncPolicy(TDynAsyncPolicy&& other) = delete;
      TDynAsyncPolicy& operator=(const TDynAsyncPolicy& other) = delete;
      TDynAsyncPolicy& operator=(TDynAsyncPolicy&& other) noexcept = delete;

      double decay_rate_{1.0};
      unsigned long repeated_action_{1};
      unsigned long n_repeated_{0};
      std::string action_topic_;
      Eigen::VectorXd desired_dynamics_;
      ros::Subscriber act_sub_;
    
      void handleActMessage(const rl_task_plugins::DesiredErrorDynamicsMsgConstPtr &act_msg);

      ros::NodeHandle nh_;

    };

} // namespace tasks

} // namespace hiqp

#endif // include guard
