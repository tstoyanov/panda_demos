#include "utilities.h"

int kdl_getQNrFromJointName(const KDL::Tree& kdl_tree, 
                            const std::string& joint_name) {
  for (auto&& it : kdl_tree.getSegments()) {
    if (it.second.segment.getJoint().getName().compare(joint_name) == 0) {
      return it.second.q_nr;
    }
  }
  return -1;
}