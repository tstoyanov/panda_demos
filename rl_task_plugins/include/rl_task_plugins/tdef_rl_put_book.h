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

#ifndef SRC_TDEF_RL_PUCK_BOOK_H
#define SRC_TDEF_RL_PUCK_BOOK_H

#include <string>

#include <hiqp/robot_state.h>
#include <hiqp/task_definition.h>
#include <kdl/treefksolverpos_recursive.hpp>
#include <kdl/treejnttojacsolver.hpp>

#include <rl_task_plugins/Act.h>
#include <ros/ros.h>
#include <ros/ros.h>
#include <vector>

namespace hiqp {
namespace tasks {

/*! \brief Represents a task definition that sets a specific joint
* configuration. The configuration can be re-set from a dedicated service call.
* This task definition does not leave any redundancy available
* to other tasks!
*  \author Quantao Yang */

class TDefRLPutBook : public TaskDefinition {
public:

    inline TDefRLPutBook() {ROS_INFO("creating object TDefRLPutBook");} //this empty constructor needed because pluginlib complains

    inline TDefRLPutBook(std::shared_ptr<GeometricPrimitiveMap> geom_prim_map,
                      std::shared_ptr<Visualizer> visualizer)
        : TaskDefinition(geom_prim_map, visualizer) {
        ROS_INFO("creating object TDefRLPutBook");
    }

    ~TDefRLPutBook() noexcept {
        ROS_INFO("Destroying object TDefRLPutBook");
    }

    int init(const std::vector<std::string>& parameters,
             RobotStatePtr robot_state);

    int update(RobotStatePtr robot_state);

    int monitor();

    //Eigen::MatrixXd getBookCorners() { return book_corners; }

private:
    TDefRLPutBook(const TDefRLPutBook& other) = delete;
    TDefRLPutBook(TDefRLPutBook&& other) = delete;
    TDefRLPutBook& operator=(const TDefRLPutBook& other) = delete;
    TDefRLPutBook& operator=(TDefRLPutBook&& other) noexcept = delete;

    std::shared_ptr<GeometricPoint> target_point_;
    KDL::Vector normal1_, normal2_, normal3_;

    Eigen::VectorXd qdot_,qddot_;

    //internal solver objects
    KDL::Frame pose_a_;        ///pose of the target_point_ frame
    KDL::Jacobian jacobian_a_; ///< tree jacobian w.r.t. the center of the frame
                                ///TDefGeometricProjection::pose_a_
    KDL::Jacobian jacobian_dot_a_;

    std::shared_ptr<KDL::TreeFkSolverPos_recursive> fk_solver_pos_;
    std::shared_ptr<KDL::TreeJntToJacSolver> fk_solver_jac_;

    // book corners
    unsigned int n_bookcorners = 4;
    std::vector<std::shared_ptr<GeometricPoint>> corner_points_;
    std::vector<KDL::Frame> pose_corners_;
    //Eigen::MatrixXd book_corners;
        
    void maskJacobian(RobotStatePtr robot_state){
        for (unsigned int c = 0; c < robot_state->getNumJoints(); ++c) {
            if (!robot_state->isQNrWritable(c)){
                J_.col(c).setZero();
            }
        }
    }

    void maskJacobianDerivative(RobotStatePtr robot_state){
        for (unsigned int c = 0; c < robot_state->getNumJoints(); ++c) {
            if (!robot_state->isQNrWritable(c)){
                J_dot_.col(c).setZero();
            }
        }
    }
};

}  // namespace tasks
}  // namespace hiqp

#endif //SRC_TDEF_RL_PUT_BOOK_H
