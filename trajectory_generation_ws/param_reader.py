import rospy

param_server_joints = "/position_joint_trajectory_controller/joints"

joint_names = rospy.get_param(param_server_joints)

print("type: ", type(joint_names))
print("type(type): ", type(joint_names[0]))
print("joint_names: ", joint_names)