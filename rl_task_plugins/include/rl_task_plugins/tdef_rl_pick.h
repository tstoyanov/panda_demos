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

#ifndef SRC_TDEF_RL_PICK_H
#define SRC_TDEF_RL_PICK_H

#include <string>

#include <hiqp/robot_state.h>
#include <hiqp/task_definition.h>
#include <kdl/treefksolverpos_recursive.hpp>
#include <kdl/treejnttojacsolver.hpp>

#include <rl_task_plugins/Act.h>
#include <ros/ros.h>
#include <ros/ros.h>


namespace hiqp {
namespace tasks {

/*! \brief Represents a task definition that sets a specific joint
* configuration. The configuration can be re-set from a dedicated service call.
* This task definition does not leave any redundancy available
* to other tasks!
*  \author Quantao Yang */

class TDefRLPick : public TaskDefinition {
public:

    inline TDefRLPick() {} //this empty constructor needed because pluginlib complains

    inline TDefRLPick(std::shared_ptr<GeometricPrimitiveMap> geom_prim_map,
                      std::shared_ptr<Visualizer> visualizer)
        : TaskDefinition(geom_prim_map, visualizer) {}

    ~TDefRLPick() noexcept {}

    int init(const std::vector<std::string>& parameters,
             RobotStatePtr robot_state);

    int update(RobotStatePtr robot_state);

    int monitor();

private:
    TDefRLPick(const TDefRLPick& other) = delete;
    TDefRLPick(TDefRLPick&& other) = delete;
    TDefRLPick& operator=(const TDefRLPick& other) = delete;
    TDefRLPick& operator=(TDefRLPick&& other) noexcept = delete;

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

#endif //SRC_TDEF_RL_PICK_H
