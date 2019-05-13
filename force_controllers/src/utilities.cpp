#include "utilities.h"
int kdl_getQNrFromJointName(const KDL::Tree& kdl_tree, 
                            const std::string& joint_name) {
  //for (auto&& it : kdl_tree.getSegments()) {
  for (KDL::SegmentMap::const_iterator it = kdl_tree.getSegments().begin(); it != kdl_tree.getSegments().end(); it++) {
    if (it->second.segment.getJoint().getName().compare(joint_name) == 0) {
      return it->second.q_nr;
    }
  }
  return -1;
}

Stiffness getParameterStiffness(ros::NodeHandle* nodeHandle)
{
    Stiffness stiffness;
    std::vector<int> translational_stiffness, rotational_stiffness;

    ROS_DEBUG("get translation/rotation stiffness");
    const size_t X = 0, Y = 1, Z = 2;
    const std::string param_translation_stiffness = "/panda/impedance_controller/cartesian_stiffness/translation";
    const std::string param_rotation_stiffness = "/panda/impedance_controller/cartesian_stiffness/rotation";

    if (!nodeHandle->getParam(param_translation_stiffness, translational_stiffness))
    {
      ROS_ERROR_STREAM("Parameter " << param_translation_stiffness << " not retreived");
    }
    if (!nodeHandle->getParam(param_rotation_stiffness, rotational_stiffness))
    {
      ROS_ERROR_STREAM("Parameter " << param_rotation_stiffness << " not retreived");
    }

    stiffness.translational_x = translational_stiffness.at(X);
    stiffness.translational_y = translational_stiffness.at(Y);
    stiffness.translational_z = translational_stiffness.at(Z);
    stiffness.rotational_x = rotational_stiffness.at(X);
    stiffness.rotational_y = rotational_stiffness.at(Y);
    stiffness.rotational_z = rotational_stiffness.at(Z);

    return stiffness;
}

Damping getParameterDamping(ros::NodeHandle* nodeHandle)
{
    Damping damping;

    std::vector<int> translational_damping, rotational_damping;

    const size_t X = 0, Y = 1, Z = 2;
    const std::string param_translation_damping = "/panda/impedance_controller/cartesian_damping/translation";
    const std::string param_rotation_damping = "/panda/impedance_controller/cartesian_damping/rotation";

    if (!nodeHandle->getParam(param_translation_damping, translational_damping))
    {
      ROS_ERROR_STREAM("Parameter " << param_translation_damping << " not retreived");
    }
    if (!nodeHandle->getParam(param_rotation_damping, rotational_damping))
    {
      ROS_ERROR_STREAM("Parameter " << param_rotation_damping << " not retreived");
    }

    damping.translational_x = translational_damping.at(X);
    damping.translational_y = translational_damping.at(Y);
    damping.translational_z = translational_damping.at(Z);
    damping.rotational_x = rotational_damping.at(X);
    damping.rotational_y = rotational_damping.at(Y);
    damping.rotational_z = rotational_damping.at(Z);

    return damping;
}
