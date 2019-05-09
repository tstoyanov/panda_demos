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

#include <limits>
#include <random>

#include <hiqp/utilities.h>

#include <rl_task_plugins/tdyn_async_policy.h>
#include <pluginlib/class_list_macros.h>

#include <ros/ros.h>

namespace hiqp
{
  namespace tasks
  {

    TDynAsyncPolicy::TDynAsyncPolicy(
      std::shared_ptr<GeometricPrimitiveMap> geom_prim_map,
      std::shared_ptr<Visualizer> visualizer)
    : TaskDynamics(geom_prim_map, visualizer) {}


      int TDynAsyncPolicy::init(const std::vector<std::string>& parameters, 
              RobotStatePtr robot_state, 
              const Eigen::VectorXd& e_initial, 
              const Eigen::VectorXd& e_dot_initial, 
              const Eigen::VectorXd& e_final, 
              const Eigen::VectorXd& e_dot_final) {

      int size = parameters.size();
      if (size != 4) {
        printHiqpWarning("TDynAsyncPolicy requires 4 parameters, got " 
          + std::to_string(size) + "! Initialization failed!");

        return -1;
      }

      damping_ = std::stod(parameters.at(1));
      action_topic_ = parameters.at(2);
      state_topic_ = parameters.at(3);

      act_sub_ = nh_.subscribe(action_topic_, 1, &TDynAsyncPolicy::handleActMessage, this);
      state_pub_ = nh_.advertise<rl_task_plugins::StateMsg>(state_topic_, 1);

      e_ddot_star_.resize(e_initial.rows());
      desired_dynamics_ = Eigen::VectorXd::Zero(e_initial.rows());

      last_publish_ = ros::Time::now();
      return 0;
    }

    int TDynAsyncPolicy::update(const RobotStatePtr robot_state, 
                const std::shared_ptr< TaskDefinition > def) {

      Eigen::VectorXd qdot=robot_state->kdl_jnt_array_vel_.qdot.data;
      e_ddot_star_= desired_dynamics_ - damping_*def->getTaskDerivative();

      publishStateMessage(def->getTaskValue());

      return 0;
    }

    int TDynAsyncPolicy::monitor() {
      return 0;
    }

    void TDynAsyncPolicy::handleActMessage(const rl_task_plugins::DesiredErrorDynamicsMsgConstPtr 
            &act_msg) {

        if(act_msg->e_ddot_star.size() != desired_dynamics_.size()) {
            printHiqpWarning("TDynAsyncPolicy: mismatch between dimensions of desired and available errors");
            std::cerr<<"des "<<desired_dynamics_.size() <<" provided "
                     <<act_msg->e_ddot_star.size()<<std::endl;
            return;
        }
        
        for(int i=0; i<act_msg->e_ddot_star.size(); ++i) {
            desired_dynamics_(i) = act_msg->e_ddot_star[i];
        }
    }

    void TDynAsyncPolicy::publishStateMessage(const Eigen::VectorXd &error) {

        ros::Time now = ros::Time::now();
        ros::Duration d = now - last_publish_;
        if(d.toSec() >= 1.0 / publish_rate_ ) {
            rl_task_plugins::StateMsg msg;
            for(int i=0; i<error.rows(); ++i) {
                msg.e.push_back(error(i));
            }
            state_pub_.publish(msg);
            last_publish_ = now;
        }

    }


} // namespace tasks

} // namespace hiqp

PLUGINLIB_EXPORT_CLASS(hiqp::tasks::TDynAsyncPolicy,
 hiqp::TaskDynamics)
