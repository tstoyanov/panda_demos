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


#include <rl_task_plugins/tdef_rl_pick.h>

#include <hiqp/utilities.h>

#include <iostream>

#include <pluginlib/class_list_macros.h>

namespace hiqp {
    namespace tasks {

        int TDefRLPick::init(const std::vector<std::string>& parameters,
                                RobotStatePtr robot_state) {
            int size = parameters.size();
            unsigned int n_controls = 3;
            unsigned int n_joints = robot_state->getNumJoints();
            if (size != 11 ) {
                printHiqpWarning("'" + getTaskName() +
                                 "': TDefRLPick expects 11 parameters ");
                return -1;
            }

            e_ = Eigen::VectorXd::Zero(n_controls);
            f_ = Eigen::VectorXd::Zero(n_controls);
            e_dot_ = Eigen::VectorXd::Zero(n_controls);
            J_ = Eigen::MatrixXd::Zero(n_controls, n_joints);
            J_dot_ = Eigen::MatrixXd::Zero(n_controls, n_joints);
            performance_measures_.resize(n_controls);
            task_signs_.insert(task_signs_.begin(), n_controls, 0); //equality task
            qddot_ = Eigen::VectorXd::Zero(n_joints);
            qdot_ = Eigen::VectorXd::Zero(n_joints);

            fk_solver_pos_ =
                    std::make_shared<KDL::TreeFkSolverPos_recursive>(robot_state->kdl_tree_);
            fk_solver_jac_ =
                    std::make_shared<KDL::TreeJntToJacSolver>(robot_state->kdl_tree_);


            normal1_ = KDL::Vector(std::stod(parameters.at(1)),std::stod(parameters.at(2)),std::stod(parameters.at(3)));
            normal2_ = KDL::Vector(std::stod(parameters.at(4)),std::stod(parameters.at(5)),std::stod(parameters.at(6)));
            normal3_ = KDL::Vector(std::stod(parameters.at(7)),std::stod(parameters.at(8)),std::stod(parameters.at(9)));

            if((KDL::dot(normal1_,normal2_) > 1e-5) ||
                (KDL::dot(normal1_,normal3_) > 1e-5) ||
                (KDL::dot(normal2_,normal3_) > 1e-5)) {
                printHiqpWarning("'" + getTaskName() +
                                 "': Task not added, TDefRLPick expects orthogonal normal vectors ");
                return -2;
            }

            //normal1_.normalize();
            //normal2_.normalize();

            std::shared_ptr<GeometricPrimitiveMap> gpm = this->getGeometricPrimitiveMap();

            target_point_ = gpm->getGeometricPrimitive<GeometricPoint>(parameters.at(10));
            if (target_point_ == nullptr) {
                printHiqpWarning(
                        "In TDefRLPick::init(), couldn't find primitive with name "
                        "'" +
                        parameters.at(10) + "'. Unable to create task!");
                return -3;
            }

            gpm->addDependencyToPrimitive(parameters.at(10), this->getTaskName());

            return 0;
        }

        int TDefRLPick::update(RobotStatePtr robot_state)
        {
            int retval = 0;

            retval = fk_solver_pos_->JntToCart(robot_state->kdl_jnt_array_vel_.q, pose_a_,
                                               target_point_->getFrameId());
            if (retval != 0) {
                std::cerr << "In TDefRLPick::update : Can't solve position "
                          << "of link '" << target_point_->getFrameId() << "'"
                          << " in the "
                          << "KDL tree! KDL::TreeFkSolverPos_recursive::JntToCart return "
                          << "error code '" << retval << "'\n";
                return -1;
            }

            jacobian_a_.resize(robot_state->kdl_jnt_array_vel_.q.rows());
            retval = fk_solver_jac_->JntToJac(robot_state->kdl_jnt_array_vel_.q,
                                              jacobian_a_, target_point_->getFrameId());
            if (retval != 0) {
                std::cerr << "In TDefRLPick::update : Can't solve jacobian "
                          << "of link '" << target_point_->getFrameId() << "'"
                          << " in the "
                          << "KDL tree! KDL::TreeJntToJacSolver return error code "
                          << "'" << retval << "'\n";
                return -3;
            }

            jacobian_dot_a_.resize(robot_state->kdl_jnt_array_vel_.q.rows());
            retval = treeJntToJacDot(robot_state->kdl_tree_, jacobian_a_, robot_state->kdl_jnt_array_vel_,
                                     jacobian_dot_a_, target_point_->getFrameId());
            if (retval != 0) {
                std::cerr << "In TDefRLPick::update : Can't solve jacobian derivative "
                          << "of link '" << target_point_->getFrameId() << "'"
                          << " in the "
                          << "KDL tree! treeJntToJacDot return error code "
                          << "'" << retval << "'\n";
                return -5;
            }

            //Note to self: the two plane normals are assumed to be in robot base frame,
            //Hence, pose is Identity and jacobians are zeros

            //calculate projection
            KDL::Vector p1__ = pose_a_.M * target_point_->getPointKDL(); //point 1 from link origin to ee
            //expressed in the world frame
            KDL::Vector p1 = pose_a_.p + p1__; //absolute ee point 1 expressed in the world frame

            double q_nr=jacobian_a_.columns();
            KDL::Jacobian J_p1k, J_dot_p1k;
            J_p1k.resize(q_nr);
            J_dot_p1k.resize(q_nr);

            changeJacRefPoint(jacobian_a_, p1__, J_p1k);
            changeJacDotRefPoint(jacobian_a_, jacobian_dot_a_, robot_state->kdl_jnt_array_vel_, p1__, J_dot_p1k);

            e_(0) = dot(normal1_,p1);
            e_(1) = dot(normal2_,p1);
            e_(2) = dot(normal3_,p1);


            J_.row(0) = Eigen::Map<Eigen::Matrix<double,1,3> >(normal1_.data)*J_p1k.data.topRows<3>();
            J_.row(1) = Eigen::Map<Eigen::Matrix<double,1,3> >(normal2_.data)*J_p1k.data.topRows<3>();
            J_.row(2) = Eigen::Map<Eigen::Matrix<double,1,3> >(normal3_.data)*J_p1k.data.topRows<3>();

            Eigen::VectorXd qdot=robot_state->kdl_jnt_array_vel_.qdot.data;
            //qddot_ = (1/robot_state->sampling_time_)*(qdot-qdot_);
            //qdot_=qdot;

            e_dot_=J_*qdot;

            J_dot_.row(0) = Eigen::Map<Eigen::Matrix<double,1,3> >(normal1_.data)*J_dot_p1k.data.topRows<3>();
            J_dot_.row(1) = Eigen::Map<Eigen::Matrix<double,1,3> >(normal2_.data)*J_dot_p1k.data.topRows<3>();
            J_dot_.row(2) = Eigen::Map<Eigen::Matrix<double,1,3> >(normal3_.data)*J_dot_p1k.data.topRows<3>();

            //FIXME seems this is not working properly
            qddot_ = robot_state->kdl_effort_.data;
            performance_measures_ = J_*qddot_ + J_dot_*qdot; // e_ddot

            //cleanup
            maskJacobian(robot_state);
            maskJacobianDerivative(robot_state);

            //awkward fix to not let the contribution of J_dot get out of hand due to numerical issues with large joint velocities induced by singularities
            double tol=1e5;
            if(fabs((J_dot_*robot_state->kdl_jnt_array_vel_.qdot.data)(0)) > tol){
                J_dot_.setZero();
                e_dot_.setZero();
            }

            return 0;
        }

        int TDefRLPick::monitor() { return 0; }

    }  // namespace tasks

}  // namespace hiqp

PLUGINLIB_EXPORT_CLASS(hiqp::tasks::TDefRLPick,
                       hiqp::TaskDefinition)