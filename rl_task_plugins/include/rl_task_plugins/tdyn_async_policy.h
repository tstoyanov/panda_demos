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
#include <rl_task_plugins/StateMsg.h>

#include <boost/thread/mutex.hpp>

namespace hiqp
{
  namespace tasks
  {

  /*! \brief Random task error behavior? 
   *  \author Jens Lundell, Todor Stoyanov */  
    class TDynAsyncPolicy : public TaskDynamics {
    public:

        inline TDynAsyncPolicy() : TaskDynamics() { ROS_INFO("creating object TDynPolicy"); initialized_=false;}
      
      TDynAsyncPolicy(std::shared_ptr<GeometricPrimitiveMap> geom_prim_map,
       std::shared_ptr<Visualizer> visualizer);

        ~TDynAsyncPolicy() noexcept { ROS_INFO("Destroying object TDynPolicy"); nh_.shutdown(); initialized_=false; }

      int init(const std::vector<std::string>& parameters, 
              RobotStatePtr robot_state, 
              const Eigen::VectorXd& e_initial, 
              const Eigen::VectorXd& e_dot_initial, 
              const Eigen::VectorXd& e_final, 
              const Eigen::VectorXd& e_dot_final);

      int update(const RobotStatePtr robot_state,
                 const  TaskDefinitionPtr def);

      int monitor();

    private:
      TDynAsyncPolicy(const TDynAsyncPolicy& other) = delete;
      TDynAsyncPolicy(TDynAsyncPolicy&& other) = delete;
      TDynAsyncPolicy& operator=(const TDynAsyncPolicy& other) = delete;
      TDynAsyncPolicy& operator=(TDynAsyncPolicy&& other) noexcept = delete;

      std::string action_topic_;
      std::string state_topic_;
      std::string logdir_base_;
      float damping_{1.0};
      unsigned int publish_rate_{100};
      ros::Time last_publish_;
      Eigen::VectorXd desired_dynamics_;
      ros::Subscriber act_sub_;
      ros::Publisher state_pub_;
      bool initialized_{false};
    
      void handleActMessage(const rl_task_plugins::DesiredErrorDynamicsMsgConstPtr &act_msg);
      void publishStateMessage(const Eigen::VectorXd &error);

      ros::NodeHandle nh_;

      boost::mutex update_lock_;

    };

} // namespace tasks

} // namespace hiqp

#endif // include guard
