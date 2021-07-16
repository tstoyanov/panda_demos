#include "impedance_controller.h"

#include <iostream>
#include <cmath>
#include <memory>
#include <string>

#include "utilities.h"

#include <geometry_msgs/Point.h>
#include <std_msgs/Float64MultiArray.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pluginlib/class_list_macros.h>
#include <eigen_conversions/eigen_kdl.h>

namespace force_controllers {

  void ImpedanceController::initialize(RobotStatePtr robot_state) {
    trajectory_subscriber_ = getControllerNodeHandle().subscribe(
      "equilibrium_pose", 1000, &ImpedanceController::trajectorySubscriberCallback, this);
    sub_desired_stiffness_ = getControllerNodeHandle().subscribe(
      "desired_stiffness", 1000, &ImpedanceController::desiredStiffnessCallback, this);

    marker_array_pub_ = getControllerNodeHandle().advertise<visualization_msgs::MarkerArray>(
      "visualization_marker", 1);

    //u_ = Eigen::VectorXd::Zero(3);
    ee_vel_ = Eigen::VectorXd::Zero(6);

    //fk_solver_pos_ = std::make_shared<KDL::TreeFkSolverPos_recursive>(robot_state->kdl_tree_);
    //fk_solver_jac_ = std::make_shared<KDL::TreeJntToJacSolver>(robot_state->kdl_tree_);
    //fk_solver_jac_dot_ = std::make_shared<KDL::ChainJntToJacDotSolver>(kdl_chain_);

    ee_jacobian_.resize(robot_state->kdl_jnt_array_vel_.q.rows());
    ee_dot_jacobian_.resize(robot_state->kdl_jnt_array_vel_.q.rows());
    
    //set initial pose to current pose
    const KDL::JntArray& q = robot_state->kdl_jnt_array_vel_.q; //FIXME value();
    fk_solver_pos_->JntToCart(q, ee_pose_, getTipName());

    Eigen::Affine3d transform;
    tf::transformKDLToEigen(ee_pose_, transform); 
    position_ = transform.translation();
    orientation_ = transform.linear();

//    tf::quaternionKDLToEigen(ee_pose_.M,orientation_);
//    tf::vectorKDLToEigen(ee_pose_.p,position_);

    position_d_ = position_;
    orientation_d_ = orientation_;
    position_d_target_ = position_;
    orientation_d_target_ = orientation_;

    q_d_nullspace_ = q.data;

    cartesian_stiffness_.setZero();
    cartesian_damping_.setZero();
    tau_prev_.setZero();
    
    Stiffness stiffness = getParameterStiffness(&getControllerNodeHandle());
    //Damping damping = getParameterDamping(&getControllerNodeHandle());

    cartesian_stiffness_(0,0) = stiffness.translational_x;
    cartesian_stiffness_(1,1) = stiffness.translational_y;
    cartesian_stiffness_(2,2) = stiffness.translational_z;
    cartesian_stiffness_(3,3) = stiffness.rotational_x;
    cartesian_stiffness_(4,4) = stiffness.rotational_y;
    cartesian_stiffness_(5,5) = stiffness.rotational_z;
    
    cartesian_damping_(0,0) = 2.0 * sqrt(cartesian_stiffness_(0,0));
    cartesian_damping_(1,1) = 2.0 * sqrt(cartesian_stiffness_(1,1));
    cartesian_damping_(2,2) = 2.0 * sqrt(cartesian_stiffness_(2,2));
    cartesian_damping_(3,3) = 2.0 * sqrt(cartesian_stiffness_(3,3));
    cartesian_damping_(4,4) = 2.0 * sqrt(cartesian_stiffness_(4,4));
    cartesian_damping_(5,5) = 2.0 * sqrt(cartesian_stiffness_(5,5));
  
    cartesian_stiffness_target_ = cartesian_stiffness_;  
    cartesian_damping_target_ = cartesian_damping_;  
  }

  void ImpedanceController::setJointAccelerations(RobotStatePtr robot_state, Eigen::VectorXd& ddq) 
  {
    const KDL::JntArray& q = robot_state->kdl_jnt_array_vel_.q; //FIXME
    const KDL::JntArray& dq = robot_state->kdl_jnt_array_vel_.qdot;
    const KDL::JntArray& effort = robot_state->kdl_effort_;

    KDL::JntArray coriolis_kdl(getNJoints()),
                  gravity_kdl(getNJoints());
    KDL::JntSpaceInertiaMatrix mass_kdl(getNJoints());


    fk_solver_pos_->JntToCart(q, ee_pose_, getTipName());
    fk_solver_jac_->JntToJac(q, ee_jacobian_, getTipName());
    //fk_solver_jac_->JntToJacDot(robot_state->kdl_jnt_array_vel_, ee_dot_jacobian_);
    JntToJacDot(robot_state->kdl_jnt_array_vel_, ee_dot_jacobian_);

    Eigen::MatrixXd J = ee_jacobian_.data.block(0,0,6,getNJoints());
    Eigen::MatrixXd dJ = ee_dot_jacobian_.data.block(0,0,6,getNJoints());
    Eigen::MatrixXd Jinv = pinv(J);
    Eigen::MatrixXd dq_eig = dq.data;
    Eigen::MatrixXd q_eig = q.data;

#if 0
    //get gravity torque
    int error_number = id_solver_->JntToGravity(q, gravity_kdl);
    //std::cerr<<"Gravity errno "<<error_number<<" value: "<<gravity_kdl.data.transpose()<<std::endl;

    error_number = id_solver_->JntToCoriolis(q, dq, coriolis_kdl);
    //std::cerr<<"Coriolis errno "<<error_number<<" value: "<<coriolis_kdl.data.transpose()<<std::endl;

    error_number = id_solver_->JntToMass(q, mass_kdl);
    //std::cerr<<"Mass errno "<<error_number<<" value:\n" <<mass_kdl.data<<std::endl;

    //convert KDL frame to eigen
    tf::quaternionKDLToEigen(ee_pose_.M,orientation_);
    tf::vectorKDLToEigen(ee_pose_.p,position_);

    // compute error to desired pose
    // position error
    Eigen::Matrix<double, 6, 1> error;
    error.head(3) << position_ - position_d_;

    // orientation error
    if (orientation_d_target_.coeffs().dot(orientation_.coeffs()) < 0.0) {
        orientation_.coeffs() << -orientation_.coeffs();
    }
    // "difference" quaternion
    Eigen::Quaterniond error_quaternion(orientation_ * orientation_d_target_.inverse());
    // convert to axis angle
    Eigen::AngleAxisd error_quaternion_angle_axis(error_quaternion);
    
    // compute "orientation error"
    error.tail(3) << error_quaternion_angle_axis.axis() * error_quaternion_angle_axis.angle();
#endif

      Eigen::Affine3d transform;
      tf::transformKDLToEigen(ee_pose_, transform); 
      position_ = transform.translation();
      orientation_ = transform.linear();

      // compute error to desired pose
      // position error
      Eigen::Matrix<double, 6, 1> error;
      error.head(3) << position_ - position_d_;

      // orientation error
      if (orientation_d_.coeffs().dot(orientation_.coeffs()) < 0.0) {
	      orientation_.coeffs() << -orientation_.coeffs();
      }
      // "difference" quaternion
      Eigen::Quaterniond error_quaternion(orientation_.inverse() * orientation_d_);
      error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
      // Transform to base frame
      error.tail(3) << -transform.linear() * error.tail(3);

    //ddq = Jinv * (-error - dJ*dq.data);
    
    // compute control
    // allocate variables
    Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7);

    // pseudoinverse for nullspace handling
    // kinematic pseuoinverse
    Eigen::MatrixXd jacobian_transpose_pinv;
    pseudoInverse(J.transpose(), jacobian_transpose_pinv);

    // Cartesian PD control with damping ratio = 1
    tau_task << J.transpose() *
        (-cartesian_stiffness_ * error - cartesian_damping_ * (J * dq_eig));
    
    // nullspace PD control with damping ratio = 1
    tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
            J.transpose() * jacobian_transpose_pinv) *
        (nullspace_stiffness_ * (q_d_nullspace_ - q_eig) -
         (2.0 * sqrt(nullspace_stiffness_)) * dq_eig);
    
    // Desired torque
    ddq = tau_task; // + tau_nullspace + coriolis_kdl.data;

    //std::cerr<<"---------------------------------------------------------------------------------\n";
    //std::cerr<<"error = "<<error.transpose()<<std::endl;
    //std::cerr<<"tau_prev = "<<tau_prev_.transpose()<<std::endl;
    ////std::cerr<<"tau_nul = "<<tau_nullspace.transpose()<<std::endl;
    ////std::cerr<<"tau_tas = "<<tau_task.transpose()<<std::endl;
    //std::cerr<<"tau_des = "<<ddq.transpose()<<std::endl;
    //ddq = saturateTorqueRate(ddq, tau_prev_);
    //std::cerr<<"tau_sat = "<<ddq.transpose()<<std::endl;
    //tau_prev_ = ddq;

    //ddq.setZero();
    //ddq = J.transpose() * (-cartesian_stiffness_*error -cartesian_damping_*J*dq.data);

#if 0
    ee_vel_ = J * dq.data;
    setPseudoController(robot_state, tp);
    ddq = Jinv * (tp.ddr_ + u_ - dJ*dq.data);
#endif

    cartesian_stiffness_ =
	    filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * cartesian_stiffness_;
    cartesian_damping_ =
	    filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * cartesian_damping_;
    nullspace_stiffness_ =
	    filter_params_ * nullspace_stiffness_target_ + (1.0 - filter_params_) * nullspace_stiffness_;
    position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
    Eigen::AngleAxisd aa_orientation_d(orientation_d_);
    Eigen::AngleAxisd aa_orientation_d_target(orientation_d_target_);
    aa_orientation_d.axis() = filter_params_ * aa_orientation_d_target.axis() +
	    (1.0 - filter_params_) * aa_orientation_d.axis();
    aa_orientation_d.angle() = filter_params_ * aa_orientation_d_target.angle() +
	    (1.0 - filter_params_) * aa_orientation_d.angle();
    orientation_d_ = Eigen::Quaterniond(aa_orientation_d);


    renderEndEffectorAndTrajectoryPoint();
  }

  void ImpedanceController::trajectorySubscriberCallback(
    const geometry_msgs::PoseStamped::ConstPtr& msg)
  {
      position_d_target_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z; 
      Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
      orientation_d_target_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
          msg->pose.orientation.z, msg->pose.orientation.w;
      if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
          orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
      }
  }

  void ImpedanceController::renderEndEffectorAndTrajectoryPoint() {

    visualization_msgs::MarkerArray marker_array;

    {
      visualization_msgs::Marker marker;
      marker.header.frame_id = getRootName();
      marker.header.stamp = ros::Time::now();
      marker.ns = "current";
      marker.id = 1;
      marker.type = visualization_msgs::Marker::ARROW;
      marker.action = visualization_msgs::Marker::ADD; 
      marker.pose.position.x = position_(0);
      marker.pose.position.y = position_(1);
      marker.pose.position.z = position_(2);
      marker.pose.orientation.x = orientation_.x();
      marker.pose.orientation.y = orientation_.y();
      marker.pose.orientation.z = orientation_.z();
      marker.pose.orientation.w = orientation_.w();
      marker.scale.x = 0.1;
      marker.scale.y = 0.01;
      marker.scale.z = 0.01;
      marker.color.r = 1.0;
      marker.color.g = 0.0;
      marker.color.b = 0.0;
      marker.color.a = 1.0;
      marker.lifetime = ros::Duration(0);
      marker_array.markers.push_back(marker);

      {
          Eigen::Quaternion<double> rotq(0.70710678118, 0, 0, 0.70710678118);
          Eigen::Quaternion<double> resq = orientation_ * rotq;
          marker.id = 2;
          marker.pose.orientation.x = resq.x();
          marker.pose.orientation.y = resq.y();
          marker.pose.orientation.z = resq.z();
          marker.pose.orientation.w = resq.w();
          marker.color.r = 0.0;
          marker.color.g = 1.0;
          marker.color.b = 0.0;
          marker.color.a = 1.0;
          marker_array.markers.push_back(marker);
      }
      {
          Eigen::Quaternion<double> rotq(0.70710678118, 0, -0.70710678118, 0);
          Eigen::Quaternion<double> resq = orientation_ * rotq;
          marker.id = 3;
          marker.pose.orientation.x = resq.x();
          marker.pose.orientation.y = resq.y();
          marker.pose.orientation.z = resq.z();
          marker.pose.orientation.w = resq.w();
          marker.color.r = 0.0;
          marker.color.g = 0.0;
          marker.color.b = 1.0;
          marker.color.a = 1.0;
          marker_array.markers.push_back(marker);
      }
    }

    {
      visualization_msgs::Marker marker;
      marker.header.frame_id = getRootName();
      marker.header.stamp = ros::Time::now();
      marker.ns = "target";
      marker.id = 1;
      marker.type = visualization_msgs::Marker::ARROW;
      marker.action = visualization_msgs::Marker::ADD; 
      marker.pose.position.x = position_d_(0);
      marker.pose.position.y = position_d_(1);
      marker.pose.position.z = position_d_(2);
      marker.pose.orientation.x = orientation_d_.x();
      marker.pose.orientation.y = orientation_d_.y();
      marker.pose.orientation.z = orientation_d_.z();
      marker.pose.orientation.w = orientation_d_.w();
      marker.scale.x = 0.1;
      marker.scale.y = 0.01;
      marker.scale.z = 0.01;
      marker.color.r = 1.0;
      marker.color.g = 0.0;
      marker.color.b = 0.0;
      marker.color.a = 1.0;
      marker.lifetime = ros::Duration(0);
      marker_array.markers.push_back(marker);
      {
          Eigen::Quaternion<double> rotq(0.70710678118, 0, 0, 0.70710678118);
          Eigen::Quaternion<double> resq = orientation_d_target_ * rotq;
          marker.id = 2;
          marker.pose.orientation.x = resq.x();
          marker.pose.orientation.y = resq.y();
          marker.pose.orientation.z = resq.z();
          marker.pose.orientation.w = resq.w();
          marker.color.r = 0.0;
          marker.color.g = 1.0;
          marker.color.b = 0.0;
          marker.color.a = 1.0;
          marker_array.markers.push_back(marker);
      }
      {
          Eigen::Quaternion<double> rotq(0.70710678118, 0, -0.70710678118, 0);
          Eigen::Quaternion<double> resq = orientation_d_target_ * rotq;
          marker.id = 3;
          marker.pose.orientation.x = resq.x();
          marker.pose.orientation.y = resq.y();
          marker.pose.orientation.z = resq.z();
          marker.pose.orientation.w = resq.w();
          marker.color.r = 0.0;
          marker.color.g = 0.0;
          marker.color.b = 1.0;
          marker.color.a = 1.0;
          marker_array.markers.push_back(marker);
      }
    }

    marker_array_pub_.publish(marker_array);
  }

  /*! \brief This code is got from Orocos KDL but is not available in ROS indigo! */
  int ImpedanceController::JntToJacDot(const KDL::JntArrayVel& q_in, 
                                              KDL::Jacobian& jdot) {
    unsigned int segmentNr = getKDLChain().getNrOfSegments();

    jdot.data.setZero();

    KDL::Twist jac_dot_k_ = KDL::Twist::Zero();
    int k = 0;
    for (unsigned int i=0; i<segmentNr; ++i) {
      if (getKDLChain().getSegment(i).getJoint().getType()!=KDL::Joint::None) {
        for (unsigned int j=0; j<getKDLChain().getNrOfJoints(); ++j) {
            jac_dot_k_ += getPartialDerivative(ee_jacobian_,j,k) * q_in.qdot(j);
        }
        jdot.setColumn(k++, jac_dot_k_);
        jac_dot_k_ = KDL::Twist::Zero();
      }
    }

    return 0;
  }

  KDL::Twist ImpedanceController::getPartialDerivative(
    const KDL::Jacobian& bs_J_ee, 
    const unsigned int& joint_idx, 
    const unsigned int& column_idx) 
  {
    int j=joint_idx;
    int i=column_idx;

    KDL::Twist jac_j = bs_J_ee.getColumn(j);
    KDL::Twist jac_i = bs_J_ee.getColumn(i);

    KDL::Twist t_djdq = KDL::Twist::Zero();

    if (j < i) {
      t_djdq.vel = jac_j.rot * jac_i.vel;
      t_djdq.rot = jac_j.rot * jac_i.rot;
    } else if (j > i) {
      t_djdq.rot = KDL::Vector::Zero();
      t_djdq.vel = -jac_j.vel * jac_i.rot;
    }else if (j == i) {
     t_djdq.rot = KDL::Vector::Zero();
     t_djdq.vel = jac_i.rot * jac_i.vel;
   }
   return t_djdq;
 }

void ImpedanceController::desiredStiffnessCallback(
    const geometry_msgs::TwistStampedConstPtr& msg) {

    //Stiffness
    cartesian_stiffness_target_(0,0) = msg->twist.linear.x;
    cartesian_stiffness_target_(1,1) = msg->twist.linear.y;
    cartesian_stiffness_target_(2,2) = msg->twist.linear.z;
    cartesian_stiffness_target_(3,3) = msg->twist.angular.x;
    cartesian_stiffness_target_(4,4) = msg->twist.angular.y;
    cartesian_stiffness_target_(5,5) = msg->twist.angular.z;

    //Damping
    cartesian_damping_target_(0,0) = 2.0 * sqrt(cartesian_stiffness_target_(0,0));
    cartesian_damping_target_(1,1) = 2.0 * sqrt(cartesian_stiffness_target_(1,1));
    cartesian_damping_target_(2,2) = 2.0 * sqrt(cartesian_stiffness_target_(2,2));
    cartesian_damping_target_(3,3) = 2.0 * sqrt(cartesian_stiffness_target_(3,3));
    cartesian_damping_target_(4,4) = 2.0 * sqrt(cartesian_stiffness_target_(4,4));
    cartesian_damping_target_(5,5) = 2.0 * sqrt(cartesian_stiffness_target_(5,5));
//    std::cerr<<"current stiffness\n"<<cartesian_stiffness_<<std::endl;
//    std::cerr<<"target stiffness\n"<<cartesian_stiffness_target_<<std::endl;
//    std::cerr<<"current damping\n"<<cartesian_damping_<<std::endl;
//    std::cerr<<"target damping\n"<<cartesian_damping_target_<<std::endl;
}

}

PLUGINLIB_EXPORT_CLASS(force_controllers::ImpedanceController, controller_interface::ControllerBase)
