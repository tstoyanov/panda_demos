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

      ROS_INFO("Creating object TDynPolicy");
      int size = parameters.size();
      if (size != 4) {
        printHiqpWarning("TDynAsyncPolicy requires 4 parameters, got "
          + std::to_string(size) + "! Initialization failed!");

        return -1;
      }

      update_lock_.lock();
      damping_ = std::stod(parameters.at(1));
      action_topic_ = parameters.at(2);
      state_topic_ = parameters.at(3);
      //logdir_base_ = parameters.at(4);

      e_ddot_star_.resize(e_initial.rows());
      desired_dynamics_ = Eigen::VectorXd::Zero(e_initial.rows());

      last_publish_ = ros::Time::now();
      initialized_ = true;

      update_lock_.unlock();

      //subscribers and publishers will be requesting locks themselves, let's make it easier on them
      act_sub_ = nh_.subscribe(action_topic_, 1, &TDynAsyncPolicy::handleActMessage, this);
      state_pub_ = nh_.advertise<rl_task_plugins::StateMsg>(state_topic_, 1);

      return 0;
    }

    int TDynAsyncPolicy::update(const RobotStatePtr robot_state,
                                const TaskDefinitionPtr def) {

      update_lock_.lock();
      if (!initialized_) {
          ROS_ERROR("TDynAsyncPolicy not initialized!");
          update_lock_.unlock();
          return -1;
      }

      Eigen::VectorXd qdot=robot_state->kdl_jnt_array_vel_.qdot.data;
      Eigen::VectorXd q=robot_state->kdl_jnt_array_vel_.q.data;

      int tasks_dim=0;
      for(auto task_status_it=robot_state->task_status_map_.begin();
          task_status_it!=robot_state->task_status_map_.end(); task_status_it++) {
          if(task_status_it->priority_ >= def->getPriority()) continue;
          //inequality tasks take one row, equality tasks take two rows
          for(int i=0; i<task_status_it->task_signs_.size(); i++) {
              tasks_dim += (task_status_it->task_signs_[i]!=0) ? 1 : 2;
          }
      }

      //constraint Jacobian
      Eigen::MatrixXd J_upper = Eigen::MatrixXd(tasks_dim, qdot.rows());
      //constraint right-hand side
      Eigen::VectorXd rhs = Eigen::VectorXd(tasks_dim);

      Eigen::MatrixXd J_lower = def->getJacobian();

      int nt = 0;
      for(auto task_status_it=robot_state->task_status_map_.begin();
            task_status_it!=robot_state->task_status_map_.end(); task_status_it++) {
            if(task_status_it->priority_ >= def->getPriority()) continue;
                for(int i=0; i<task_status_it->task_signs_.size(); i++) {
                    int task_sign=task_status_it->task_signs_[i];
                    //std::cerr<<"i = "<<i<<" nt = "<<nt<<" s = "<<task_sign;
                    if(task_sign!=0) {
                        J_upper.row(nt) = ((double) task_sign)*task_status_it->J_.row(i);
                        rhs(nt) = ((double) task_sign)*(task_status_it->dde_star_(i)-task_status_it->dJ_.row(i).dot(qdot));
                        //std::cerr<<" J="<<task_status_it->J_.row(i)<<" dde="<<task_status_it->dde_star_(i)<<" dJ="<<task_status_it->dJ_.row(i)<<" rhs="<<rhs(nt)<<std::endl;
                        nt++;
                    } else {
                        //std::cerr<<" equality!!!\n";
                        J_upper.row(nt) = task_status_it->J_.row(i);
                        rhs(nt) = (task_status_it->dde_star_(i)-task_status_it->dJ_.row(i).dot(qdot));
                        J_upper.row(nt+1) = (-1)*task_status_it->J_.row(i);
                        rhs(nt+1) = (-1)*(task_status_it->dde_star_(i)-task_status_it->dJ_.row(i).dot(qdot));
                        nt+=2;
                    }
                }
        }

      e_ddot_star_= desired_dynamics_ - damping_*def->getTaskDerivative();

#if 0
      logdir_base_ = "/home/quantao/hiqp_logs/";
      std::ofstream J_up_stream, rhs_stream, J_stream, desired_stream;
      J_up_stream.open(logdir_base_+"J_upper.dat", std::ios::out|std::ios::app);
      J_stream.open(logdir_base_+"J_lower.dat", std::ios::out|std::ios::app);
      rhs_stream.open(logdir_base_+"rhs.dat", std::ios::out|std::ios::app);
      desired_stream.open(logdir_base_+"desired.dat", std::ios::out|std::ios::app);

      J_up_stream<<J_upper<<std::endl;
      J_stream<<J_lower<<std::endl;
      rhs_stream<<rhs.transpose()<<std::endl;
      desired_stream<<(e_ddot_star_-def->getJacobianDerivative()*q).transpose()<<std::endl;

      J_up_stream.close();
      J_stream.close();
      rhs_stream.close();
      desired_stream.close();
#endif

        //publishStateMessage(def->getTaskValue());
        Eigen::VectorXd error = def->getTaskValue();
        Eigen::VectorXd error_derivative = def->getTaskDerivative();
        Eigen::VectorXd rhs_fixed_term = (-def->getJacobianDerivative()*qdot);

        //we will do this directly here, too much to carry around right now
        ros::Time now = ros::Time::now();
        ros::Duration d = now - last_publish_;
        if(d.toSec() >= 1.0 / publish_rate_ ) {
            rl_task_plugins::StateMsg msg;
            msg.n_joints = q.rows();
            msg.n_constraints_upper = tasks_dim;
            msg.n_constraints_lower = error.rows();

            //followed by shameless deep copy
            msg.e = std::vector<double>(error.data(),error.data()+error.size());
            msg.de = std::vector<double>(error_derivative.data(), error_derivative.data()+error_derivative.size());
            msg.q = std::vector<double>(q.data(), q.data()+q.size());
            msg.dq = std::vector<double>(qdot.data(), qdot.data()+qdot.size());
            msg.ddq_star = std::vector<double>(robot_state->ddq_star.data(), robot_state->ddq_star.data()+robot_state->ddq_star.size());
            msg.J_upper = std::vector<double>(J_upper.data(), J_upper.data()+J_upper.size());
            msg.J_lower = std::vector<double>(J_lower.data(), J_lower.data()+J_lower.size());
            msg.b_upper = std::vector<double>(rhs.data(), rhs.data()+rhs.size());
            msg.rhs_fixed_term = std::vector<double>(rhs_fixed_term.data(),rhs_fixed_term.data()+rhs_fixed_term.size());

            state_pub_.publish(msg);
            last_publish_ = now;
        }
      update_lock_.unlock();

      return 0;
    }

    int TDynAsyncPolicy::monitor() {
      return 0;
    }

    void TDynAsyncPolicy::handleActMessage(const rl_task_plugins::DesiredErrorDynamicsMsgConstPtr 
            &act_msg) {

        update_lock_.lock();
        if (!initialized_) {
            ROS_ERROR("TDynAsyncPolicy not initialized!");
            update_lock_.unlock();
            return;
        }

        if(act_msg->e_ddot_star.size() != e_ddot_star_.rows()) {
            printHiqpWarning("TDynAsyncPolicy: mismatch between dimensions of eddot_star and dynamics provided in message");
            std::cerr<<" eddot_star "<<e_ddot_star_.size() <<" provided "
                     <<act_msg->e_ddot_star.size()<<std::endl;
            for (int j = 0; j < act_msg->e_ddot_star.size(); ++j) {
                std::cout << "act_msg->e_ddot_star:" << act_msg->e_ddot_star[j] << std::endl;
            }
            update_lock_.unlock();
            return;
        }

        if(act_msg->e_ddot_star.size() != desired_dynamics_.rows()) {
            printHiqpWarning("TDynAsyncPolicy: mismatch between dimensions of desired and available errors, resizing");
            desired_dynamics_ = Eigen::VectorXd::Zero(e_ddot_star_.rows());
        }
        
        for(int i=0; i<act_msg->e_ddot_star.size(); ++i) {
            desired_dynamics_(i) = act_msg->e_ddot_star[i];
        }
        update_lock_.unlock();

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
