
rostopic pub -1 /gripper_direct_controller/command std_msgs/Float64MultiArray "layout:
  dim:
  - label: ''
    size: 0
    stride: 0
  data_offset: 0
data: [0.005, 0.005]" 

sleep 3

rosservice call /hiqp_joint_velocity_controller/remove_tasks \
"names:
- 'approach_align_z'
- 'ee_goal'
" 
rosservice call /hiqp_joint_velocity_controller/remove_all_primitives "{}"


####################### GEOMETRIC PRIMITIVES #######################
rosservice call /hiqp_joint_velocity_controller/set_primitives \
"primitives:
- name: 'lift_plane'
  type: 'plane'
  frame_id: 'panda_link0'
  visible: true
  color: [1.0, 0.0, 1.0, 1.0]
  parameters: [0.0, 0.0, 1.0, 0.2]
"


####################### TASKS #######################
rosservice call /hiqp_joint_velocity_controller/set_tasks \
"tasks:
- name: 'ee_plane_lift'
  priority: 3
  visible: 1
  active: 1
  monitored: 1
  def_params: ['TDefGeomProj', 'point', 'plane', 'ee_point > lift_plane']
  dyn_params: ['TDynLinear', '4.0']
"


#- name: 'ee_rl'
#  priority: 4
#  visible: 1
#  active: 1
#  monitored: 0
#  def_params: ['TDefRLPick', '1','0','0', '0','1','0', '0','0','1', 'ee_point']
#  dyn_params: ['TDynAsyncPolicy', '1000.0', 'ee_rl/act', 'ee_rl/state']

#- name: 'approach_align_x'
#  priority: 4
#  visible: 1
#  active: 1
#  monitored: 1
#  def_params: ['TDefGeomAlign', 'line', 'line', 'ee_x_axis = world_x_axis']
#  dyn_params: ['TDynPD', '16.0', '50.0']


#- name: 'wrist_plane_project'
#  priority: 1
#  visible: 1
#  active: 1
#  monitored: 1
#  def_params: ['TDefGeomProj', 'point', 'plane', 'wrist_point > table_plane']
#  dyn_params: ['TDynPD', '4.0', '5.0']



#- name: 'elbow_plane_top'
#  priority: 1
#  visible: 1
#  active: 1
#  monitored: 1
#  def_params: ['TDefGeomProj', 'point', 'plane', 'elbow_point > top_plane']
#  dyn_params: ['TDynPD', '4.0', '5.0']






