#ifndef HELPERS_H
#define HELPERS_H

#include "ros/console.h"
#include "ros/time.h"

#include <Eigen/Geometry>

namespace Helpers
{
    Eigen::Affine3d rotateMatrixRPY(Eigen::Affine3d tMatrix, double rollAngle,
                                    double pitchAngle, double yawAngle)
    {
        Eigen::AngleAxisd roll(rollAngle, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitch(pitchAngle, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yaw(yawAngle, Eigen::Vector3d::UnitZ());

        Eigen::Quaterniond quaternion = yaw * pitch * roll;

        Eigen::Affine3d rotated_tMatrix = tMatrix.rotate(quaternion);

        return rotated_tMatrix;
    }

    void sleepAndTell(double sleepTime)
    {
        ROS_DEBUG("Sleeping for %lf seconds", sleepTime);
        ros::Duration(sleepTime).sleep();
    }

}

#endif
