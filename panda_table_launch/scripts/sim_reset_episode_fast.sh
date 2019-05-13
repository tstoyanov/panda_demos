rosservice call /hiqp_joint_effort_controller/deactivate_task "name: 'ee_rl'"
rostopic pub /ee_rl/act rl_task_plugins/DesiredErrorDynamicsMsg "e_ddot_star: [0.0, 0.0]" --once &
sleep 2
rosservice call /hiqp_joint_effort_controller/activate_task "name: 'ee_rl'"
