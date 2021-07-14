rostopic pub -1 /position_joint_trajectory_controller/command trajectory_msgs/JointTrajectory \
"header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
joint_names:
- 'panda_joint1'
- 'panda_joint2'
- 'panda_joint3'
- 'panda_joint4'
- 'panda_joint5'
- 'panda_joint6'
- 'panda_joint7' 
points:
- positions: [1.1, -0.1, 0, -1.75, 0, 1.8, 0.8]
  velocities: [0, 0, 0, 0, 0, 0, 0]
  accelerations: [0, 0, 0, 0, 0, 0, 0]
  effort: [0, 0, 0, 0, 0, 0, 0]
  time_from_start: {secs: 3, nsecs: 0}" 

sleep 3

rosservice call /controller_manager/load_controller "name: 'impedance_controller'" 
sleep 1

rosservice call /controller_manager/switch_controller \
"start_controllers:
- 'impedance_controller'
stop_controllers:
- 'position_joint_trajectory_controller'
strictness: 3" 
sleep 3

rostopic pub -r 5 /impedance_controller/equilibrium_pose geometry_msgs/PoseStamped \
"header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
pose:
  position:
    x: 0.475
    y: 0.105
    z: 0.74
  orientation:
    x: 1.0
    y: 0.0
    z: 0.0
    w: 0.0" 

