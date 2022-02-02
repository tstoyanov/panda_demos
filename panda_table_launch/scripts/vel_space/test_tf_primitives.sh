rosservice call /hiqp_joint_velocity_controller/remove_tasks \
"names:
- 'full_pose'
- 'ee_goal_p'
- 'ee_goal_a'
" 
rosservice call /hiqp_joint_velocity_controller/remove_all_primitives "{}"


####################### GEOMETRIC PRIMITIVES #######################
rosservice call /hiqp_joint_velocity_controller/set_primitives \
"primitives:
- name: 'ee_point'
  type: 'point'
  frame_id: 'panda_hand'
  visible: true
  color: [0.0, 0.0, 1.0, 1.0]
  parameters: [0.0, 0.0, 0.1]
- name: 'goal_point'
  type: 'point'
  frame_id: 'test_link'
  visible: true
  color: [0.0, 0.0, 1.0, 1.0]
  parameters: [0.1, 0.1, 0.1]
- name: 'ee_frame'
  type: 'frame'
  frame_id: 'panda_hand'
  visible: true
  color: [0.0, 0.0, 1.0, 1.0]
  parameters: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] 
"
#x yz qw qx qy qz

rosrun tf static_transform_publisher 0.3 -0.1 0.1 3.14 0 1.57 panda_link0 test_link 100 &

sleep 1

####################### TASKS #######################
rosservice call /hiqp_joint_velocity_controller/set_tasks \
"tasks:
- name: 'ee_goal_p'
  priority: 0
  visible: 1
  active: 1
  monitored: 1
  def_params: ['TDefGeomProj', 'frame', 'frame', 'ee_frame = test_link']
  dyn_params: ['TDynPD', '4.0', '5.0']
- name: 'ee_goal_a'
  priority: 0
  visible: 1
  active: 1
  monitored: 1
  def_params: ['TDefGeomAlign', 'frame', 'frame', 'ee_frame = test_link', '0']
  dyn_params: ['TDynPD', '4.0', '5.0']
- name: 'full_pose'
  priority: 2
  visible: 1
  active: 1
  monitored: 0
  def_params: ['TDefFullPose', '-0.51', '-1.17', '0.0', '-2.89', '-0.0', '1.82', '0.84']
  dyn_params: ['TDynPD', '1.0', '2.0']
"



