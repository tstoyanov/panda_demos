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

#include <rl_task_plugins/tdef_rl_full_pose.h>

#include <hiqp/utilities.h>

#include <iostream>

#include <pluginlib/class_list_macros.h>


namespace hiqp {
  namespace tasks {

    int TDefFullPoseRL::init(const std::vector<std::string>& parameters,
			   RobotStatePtr robot_state) {
      int size = parameters.size();
      unsigned int n_controls = robot_state->getNumControls();
      unsigned int n_joints = robot_state->getNumJoints();
      if (size != 1 ) {
	printHiqpWarning("TDefFullPoseRL does not accept parameters");
	return -1;
      }

      desired_configuration_ = std::vector<double>(n_controls, 0);

      q_ddot_desired_ =  Eigen::VectorXd::Zero(n_controls);
      e_ = Eigen::VectorXd::Zero(n_controls);
      f_ = Eigen::VectorXd::Zero(n_controls);      
      e_dot_ = Eigen::VectorXd::Zero(n_controls);
      J_ = Eigen::MatrixXd::Zero(n_controls, n_joints);
      J_dot_ = Eigen::MatrixXd::Zero(n_controls, n_joints);
      performance_measures_.resize(0);
      task_signs_.insert(task_signs_.begin(), n_controls, 0); //equality task
  
      q_desired_ = Eigen::Matrix<double,7,1> (robot_state->kdl_jnt_array_vel_.q.data);

      // The jacobian is constant with zero-columns for non-writable joints
      // -1  0  0  0  0
      //  0 -1  0  0  0
      //  0  0  0 -1  0
      for (int c = 0, r = 0; c < n_joints; ++c) {
	if (robot_state->isQNrWritable(c)) {
	  J_(r, c) = -1;
	  r++;
	}
      }
      
      action_client_ = nh_.advertiseService(this->getTaskName()+"/act",&TDefFullPoseRL::updateAction,this);
      return 0;
    }

    int TDefFullPoseRL::update(RobotStatePtr robot_state) {
      const KDL::JntArray& q = robot_state->kdl_jnt_array_vel_.q;
      const KDL::JntArray& q_dot = robot_state->kdl_jnt_array_vel_.qdot;
      ////forward integration////
      Eigen::Matrix<double,7,1> q_eig (q.data);
      Eigen::Matrix<double,7,1> qd_eig (q_dot.data);
      double dt = robot_state->sampling_time_;
      if(action_updated_) {
          q_desired_ = Eigen::Matrix<double,7,1> (robot_state->kdl_jnt_array_vel_.q.data);
          action_updated_ = false;
      }
      q_desired_ += dt*(qd_eig + dt*(q_ddot_desired_));
      /////////////////////////

      int j = 0;
      for (int i = 0; i < q.rows(); ++i) {
	if (robot_state->isQNrWritable(i)) {
	  //e_(j) = action_updated_ ? desired_configuration_.at(j) - q(i) : 0; //position target
	  //e_dot_(j)=action_updated_ ? -q_dot(i) : 0;
	  e_(j) = q_desired_(j) - q(i);
	  e_dot_(j)=-q_dot(i);
	  j++;
	}
      }
      return 0;
    }

    int TDefFullPoseRL::monitor() { return 0; }

    bool TDefFullPoseRL::updateAction(rl_task_plugins::Act::Request &req, 
            rl_task_plugins::Act::Response &res) {
        if(req.qd.size() != desired_configuration_.size()) {
            printHiqpWarning("TDefFullPoseRL:: Can't set new action due to dimensionality mismatch!");
            return true;
        }
        //desired_configuration_ = req.qd; //position target
        q_ddot_desired_ = Eigen::Matrix<double,7,1> (req.qd.data()); //ugly
        action_updated_ = true;

        return true;
    }

  }  // namespace tasks

}  // namespace hiqp

PLUGINLIB_EXPORT_CLASS(hiqp::tasks::TDefFullPoseRL,
                               hiqp::TaskDefinition)

