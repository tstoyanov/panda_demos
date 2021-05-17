#!/bin/bash
source /home/quantao/Workspaces/catkin_ws/devel/setup.bash
echo $ROS_PACKAGE_PATH
for seed in 123; do # 123 231 312; do
  for u_step in 10; do
    for noise in 2.0; do
      for runid in 1; do
       /home/quantao/anaconda3/envs/py37/bin/python main.py --env PandaEnv --algo NAF --seed ${seed} --num_episodes 1000 --run_id ${runid} --noise_scale ${noise} --logdir \
/home/quantao/hiqp_logs/PandaReaching/NAF/ --updates_per_step ${u_step} --exploration_end 50
#        /home/quantao/anaconda3/envs/py37/bin/python main.py --env PandaEnv --algo DDPG --seed ${seed} --num_episodes 200  --run_id ${runid} --noise_scale ${noise} --logdir \
#/home/quantao/hiqp_logs/PandaReaching/DDPG/ --updates_per_step ${u_step} --exploration_end 50
#        /home/quantao/anaconda3/envs/py37/bin/python train_reinforce.py --env PandaEnv --algo REINFORCE --seed ${seed} --num_episodes 200  --run_id ${runid} --noise_scale ${noise} --logdir \
#/home/quantao/hiqp_logs/PandaReaching/REINFORCE/ --updates_per_step ${u_step}
      done
    done
  done
done
