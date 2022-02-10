#ifndef IMPEDANCE_CONTROLLER_H
#define IMPEDANCE_CONTROLLER_H

#include "base_effort_controller.h"

//#include <atomic>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <ros/ros.h>
#include <ros/node_handle.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>

#include <kdl/jacobian.hpp>
#include <kdl/jntarrayvel.hpp>
//#include <kdl/chainjnttojacdotsolver.hpp>

namespace force_controllers {

  /*! \brief A joint effort controller that mimics a mass-spring-damper system following a trajectory.
   *  \author Marcus A Johansson */
  class ImpedanceController : public BaseEffortController {
  public:
    ImpedanceController() {}
    ~ImpedanceController() {}

    void initialize(RobotStatePtr robot_state);

    void setJointAccelerations(RobotStatePtr robot_state, Eigen::VectorXd& ddq);

  private:
    ImpedanceController(const ImpedanceController& other) = delete;
    ImpedanceController(ImpedanceController&& other) = delete;
    ImpedanceController& operator=(const ImpedanceController& other) = delete;
    ImpedanceController& operator=(ImpedanceController&& other) noexcept = delete;

    ros::Subscriber                       trajectory_subscriber_;
    void trajectorySubscriberCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
  
    // Stiffness profile subscriber
    ros::Subscriber sub_desired_stiffness_;
    void desiredStiffnessCallback(const geometry_msgs::TwistStampedConstPtr& msg);

    void renderEndEffectorAndTrajectoryPoint();

    int JntToJacDot(const KDL::JntArrayVel& q_in, KDL::Jacobian& jdot);

    KDL::Twist getPartialDerivative(const KDL::Jacobian& bs_J_ee, 
                                    const unsigned int& joint_idx, 
                                    const unsigned int& column_idx);

    double filter_params_{0.005};

    double nullspace_stiffness_{1.0};
    double nullspace_stiffness_target_{1.0};

    Eigen::Vector3d     position_d_target_;
    Eigen::Quaterniond  orientation_d_target_;
    Eigen::Vector3d     position_d_;
    Eigen::Quaterniond  orientation_d_;
    Eigen::Vector3d     position_;
    Eigen::Quaterniond  orientation_;
    Eigen::Matrix<double, 7, 1> q_d_nullspace_;
    Eigen::Matrix<double, 7, 1> tau_prev_;

    Eigen::Matrix<double, 6, 6> cartesian_stiffness_;
    Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_;
    Eigen::Matrix<double, 6, 6> cartesian_damping_;
    Eigen::Matrix<double, 6, 6> cartesian_damping_target_;

    //Eigen::VectorXd                       u_; // pseudo controls

    KDL::Frame                            ee_pose_; // end-effector pose
    Eigen::VectorXd                       ee_vel_; // end-effector velocity
    KDL::Jacobian                         ee_jacobian_; // end-effector jacobian
    KDL::Jacobian                         ee_dot_jacobian_; // end-effector time derivative of jacobian

    //std::shared_ptr<KDL::ChainJntToJacDotSolver>     fk_solver_jac_dot_; // forward kinematics end-effector time derivative of jacobian solver

    ros::Publisher                        marker_array_pub_;

    const double delta_tau_max_{1.0}; 

    Eigen::Matrix<double, 7, 1> saturateTorqueRate(
    	const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    	const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  	
	    Eigen::Matrix<double, 7, 1> tau_d_saturated{};
	    for (size_t i = 0; i < 7; i++) {
		    double difference = tau_d_calculated[i] - tau_J_d[i];
		    tau_d_saturated[i] =
			    tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
	    }
	    return tau_d_saturated;
    }

  };

  inline void pseudoInverse(const Eigen::MatrixXd& M_, Eigen::MatrixXd& M_pinv_, bool damped = true) {
      double lambda_ = damped ? 0.2 : 0.0;

      Eigen::JacobiSVD<Eigen::MatrixXd> svd(M_, Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType sing_vals_ = svd.singularValues();
      Eigen::MatrixXd S_ = M_;  // copying the dimensions of M_, its content is not needed.
      S_.setZero();

      for (int i = 0; i < sing_vals_.size(); i++)
          S_(i, i) = (sing_vals_(i)) / (sing_vals_(i) * sing_vals_(i) + lambda_ * lambda_);

      M_pinv_ = Eigen::MatrixXd(svd.matrixV() * S_.transpose() * svd.matrixU().transpose());
  }

}

#endif
