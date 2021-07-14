rosservice call /hiqp_joint_velocity_controller/remove_tasks \
"names:
- 'full_pose'
- 'ee_plane_project'
- 'approach_align_z'
- 'ee_cage_front'
- 'ee_cage_back'
- 'ee_cage_left'
- 'ee_cage_right'
- 'ee_goal'
" 
rosservice call /hiqp_joint_velocity_controller/remove_all_primitives "{}"

#rosservice call /hiqp_joint_velocity_controller/set_tasks \
#"tasks:
#- name: 'full_pose'
#  priority: 5
#  visible: 1
#  active: 1
#  monitored: 0
#  def_params: ['TDefFullPose', '1.51', '-1.17', '0.0', '-2.89', '-0.0', '1.82', '0.84']
#  dyn_params: ['TDynPD', '9.0', '7.0']"
#
#rosservice call /gazebo/unpause_physics "{}"

#sleep 2


####################### GEOMETRIC PRIMITIVES #######################
rosservice call /hiqp_joint_velocity_controller/set_primitives \
"primitives:
- name: 'ee_point'
  type: 'point'
  frame_id: 'panda_hand'
  visible: true
  color: [0.0, 0.0, 1.0, 1.0]
  parameters: [0.0, 0.0, 0.1]
- name: 'elbow_point'
  type: 'point'
  frame_id: 'panda_link4'
  visible: true
  color: [1.0, 0.0, 0.0, 1.0]
  parameters: [0.0, 0.0, 0.0]
- name: 'ee_x_axis'
  type: 'line'
  frame_id: 'panda_hand'
  visible: true
  color: [1.0, 0.0, 0.0, 1.0]
  parameters: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- name: 'ee_z_axis'
  type: 'line'
  frame_id: 'panda_hand'
  visible: true
  color: [0.0, 1.0, 1.0, 1.0]
  parameters: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
- name: 'table_plane'
  type: 'plane'
  frame_id: 'world'
  visible: true
  color: [1.0, 0.0, 1.0, 1.0]
  parameters: [0.0, 0.0, 1.0, 0.77]
- name: 'world_x_axis'
  type: 'line'
  frame_id: 'world'
  visible: true
  color: [1.0, 0.0, 0.0, 1.0]
  parameters: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- name: 'table_z_axis'
  type: 'line'
  frame_id: 'world'
  visible: true
  color: [0.0, 1.0, 1.0, 1.0]
  parameters: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
- name: back_plane
  type: plane
  frame_id: world
  visible: false
  color: [0.2, 0.5, 0.2, 0.21]
  parameters: [0.0, 1.0, 0.0, -0.3]
- name: front_plane
  type: plane
  frame_id: world
  visible: false
  color: [0.2, 0.5, 0.2, 0.21]
  parameters: [0.0, 1.0, 0.0, 0.2]
- name: left_plane
  type: plane
  frame_id: world
  visible: false
  color: [0.2, 0.5, 0.2, 0.21]
  parameters: [1.0, 0.0, 0.0, 0.1]
- name: right_plane
  type: plane
  frame_id: world
  visible: false
  color: [0.2, 0.5, 0.2, 0.21]
  parameters: [1.0, 0.0, 0.0, -0.4]
- name: 'goal'
  type: 'box'
  frame_id: 'world'
  visible: true
  color: [0.0, 1.0, 0.0, 1.0]
  parameters: [-0.2, -0.0, 0.79, 0.04, 0.04, 0.04]
- name: 'goal_point'
  type: 'point'
  frame_id: 'world'
  visible: true
  color: [0.0, 0.0, 1.0, 1.0]
  parameters: [-0.2, -0.0, 0.79]
"


####################### TASKS #######################
rosservice call /hiqp_joint_velocity_controller/set_tasks \
"tasks:
- name: 'ee_plane_table'
  priority: 1
  visible: 1
  active: 1
  monitored: 1
  def_params: ['TDefGeomProj', 'point', 'plane', 'ee_point > table_plane']
  dyn_params: ['TDynPD', '4.0', '5.0']
- name: 'ee_cage_front'
  priority: 2
  visible: 1
  active: 1
  monitored: 1
  def_params: ['TDefGeomProj', 'point', 'plane', 'ee_point < front_plane']
  dyn_params: ['TDynPD', '4.0', '5.0']
- name: 'ee_cage_back'
  priority: 2
  visible: 1
  active: 1
  monitored: 1
  def_params: ['TDefGeomProj', 'point', 'plane', 'ee_point > back_plane']
  dyn_params: ['TDynPD', '4.0', '5.0']
- name: 'ee_cage_left'
  priority: 2
  visible: 1
  active: 1
  monitored: 1
  def_params: ['TDefGeomProj', 'point', 'plane', 'ee_point < left_plane']
  dyn_params: ['TDynPD', '4.0', '5.0']
- name: 'ee_cage_right'
  priority: 2
  visible: 1
  active: 1
  monitored: 1
  def_params: ['TDefGeomProj', 'point', 'plane', 'ee_point > right_plane']
  dyn_params: ['TDynPD', '4.0', '5.0']
- name: 'approach_align_z'
  priority: 3
  visible: 1
  active: 1
  monitored: 1
  def_params: ['TDefGeomAlign', 'line', 'line', 'ee_z_axis = table_z_axis']
  dyn_params: ['TDynPD', '9.0', '7.0']
- name: 'ee_goal'
  priority: 4
  visible: 1
  active: 1
  monitored: 1
  def_params: ['TDefGeomProj', 'point', 'point', 'ee_point = goal_point']
  dyn_params: ['TDynPD', '4.0', '5.0']
- name: 'full_pose'
  priority: 5
  visible: 1
  active: 1
  monitored: 0
  def_params: ['TDefFullPose', '1.51', '-1.17', '0.0', '-2.89', '-0.0', '1.82', '0.84']
  dyn_params: ['TDynPD', '9.0', '7.0']
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






