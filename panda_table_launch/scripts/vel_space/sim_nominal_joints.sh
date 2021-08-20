rosservice call /hiqp_joint_velocity_controller/set_tasks \
"tasks:
- name: 'nominal_joints'
  priority: 0
  visible: 1
  active: 1
  monitored: 1
  def_params: ['TDefFullPose', '1.51', '-1.17', '0.003', '-2.89', '-0.0', '1.32', '0.84']
  dyn_params: ['TDynLinear', '10.5'] "
