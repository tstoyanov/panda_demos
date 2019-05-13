#!/usr/bin/env python
import rospy
from rl_task_plugins.msg import DesiredErrorDynamicsMsg
from rl_task_plugins.msg import StateMsg

import argparse
import matplotlib.pyplot as plt
import gym
import numpy as np
import torch
import time
import subprocess

#import files...
from naf import NAF
from ounoise import OUNoise
from replay_memory import Transition, ReplayMemory
from environment import Env

subdata = []


def callback(data):
    global subdata
    subdata = data.e


rospy.init_node('DRL_traffic', anonymous=True)


def sim_reset_start():
    subprocess.call("~/catkin_workspace/src/panda_demos/panda_table_launch/scripts/sim_reset_episode.sh", shell=True)
    subprocess.call("~/catkin_workspace/src/panda_demos/panda_table_launch/scripts/sim_2drl_tasks.sh", shell=True)


def sim_reset():
    subprocess.call("~/catkin_workspace/src/panda_demos/panda_table_launch/scripts/sim_reset_episode_fast.sh", shell=True)


def main():
    global subdata
    t_start = time.time()

    parser = argparse.ArgumentParser(description='PyTorch X-job')
    parser.add_argument('--env_name', default="OurEnv-v0",
                        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.001,
                        help='discount factor for model (default: 0.001)')
    parser.add_argument('--ou_noise', type=bool, default=True)
    parser.add_argument('--noise_scale', type=float, default=0.4, metavar='G',
                        help='initial noise scale (default: 0.3)')
    parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                        help='final noise scale (default: 0.4)')
    parser.add_argument('--exploration_end', type=int, default=20, metavar='N',
                        help='number of episodes with noise (default: 100)')
    parser.add_argument('--seed', type=int, default=4, metavar='N',
                        help='random seed (default: 4)')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='batch size (default: 512)')
    parser.add_argument('--num_steps', type=int, default=300, metavar='N',
                        help='max episode length (default: 1000)')
    parser.add_argument('--num_episodes', type=int, default=500, metavar='N',
                        help='number of episodes (default: 1000)')
    parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                        help='hidden size (default: 128)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 1000000)')
    parser.add_argument('--save_agent', type=bool, default=True,
                        help='save model to file')
    parser.add_argument('--load_agent', type=bool, default=False,
                        help='load model from file')
    parser.add_argument('--train_model', type=bool, default=True,
                        help='Training or run')
    parser.add_argument('--greedy_steps', type=int, default=5, metavar='N',
                        help='amount of times greedy goes (default: 100)')

    args = parser.parse_args()

    #env = gym.make(args.env_name)

    env = Env()

    #env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # -- initialize agent, Q and Q' --
    agent = NAF(args.gamma, args.tau, args.hidden_size,
                env.observation_space.shape[0], env.action_space)

    # -- load existing model --
    if args.load_agent:
        agent.load_model(args.env_name, args.batch_size, '.pth')
        print("agent: naf_{}_{}_{}, is loaded").format(args.env_name, args.batch_size, '.pth')

    # -- declare memory buffer and random process N
    memory = ReplayMemory(args.replay_size)
    ounoise = OUNoise(env.action_space.shape[0]) if args.ou_noise else None

    rewards = []
    total_numsteps = 0
    greedy_reward = []
    avg_greedy_reward = []
    upper_reward = []
    lower_reward = []

    sim_reset_start()

    pub = rospy.Publisher('/ee_rl/act', DesiredErrorDynamicsMsg, queue_size=10)
    rospy.Subscriber("/ee_rl/state", StateMsg, callback)
    rate = rospy.Rate(9)
    rate.sleep()
    for i_episode in range(args.num_episodes):
        # -- reset environment for every episode --
        sim_reset()
        state = torch.Tensor(subdata).unsqueeze(0)

        # -- initialize noise (random process N) --
        if args.ou_noise:
            ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(
                0, args.exploration_end - i_episode / args.exploration_end + args.final_noise_scale)
            ounoise.reset()

        episode_reward = 0

        while True:
            # -- action selection, observation and store transition --
            #action = agent.select_action(state)
            action = agent.select_action(state, ounoise)
            a = action.numpy()[0] * 50
            act_pub = [a[0], a[1]]
            #print(action.numpy()[0] * 50)
            print(act_pub)
            pub.publish(act_pub)
            next_state = torch.Tensor(subdata).unsqueeze(0)
            reward, done = env.calc_shaped_reward(next_state)

            total_numsteps += 1
            episode_reward += reward

            action = torch.Tensor(action)
            mask = torch.Tensor([not done])
            reward = torch.Tensor([reward])

            memory.push(state, action, mask, next_state, reward)

            state = next_state

            # -- training --
            # print("len(memory): {}".format(len(memory)))
            if len(memory) > args.batch_size and args.train_model:
                for _ in range(20):
                    transitions = memory.sample(args.batch_size)
                    batch = Transition(*zip(*transitions))

                    agent.update_parameters(batch)
            else:
                time.sleep(0.1)
            rate.sleep()

            if done or total_numsteps % args.num_steps == 0:
                break

        pub.publish([0, 0])
        rewards.append(episode_reward)

        if args.train_model:
            greedy_episode = max(args.num_episodes/100, 10)
        else:
            greedy_episode = 10
        greedy_range = min(args.greedy_steps, greedy_episode)

        # -- calculates episode without noise --
        if i_episode % greedy_episode == 0 and not i_episode == 0:
            for _ in range(0, greedy_range):
                # -- reset environment for every episode --
                sim_reset()

                state = torch.Tensor(subdata).unsqueeze(0)
                episode_reward = 0
                steps = 0

                while True:
                    action = agent.select_action(state)
                    pub.publish(action.numpy()[0] * 50)
                    next_state = torch.Tensor(subdata).unsqueeze(0)
                    reward, done = env.calc_shaped_reward(next_state)
                    episode_reward += reward

                    state = next_state
                    steps += 1
                    if done or steps == args.num_steps:
                        greedy_reward.append(episode_reward)
                        break
                    rate.sleep()

            upper_reward.append(np.max(rewards[-greedy_episode:]))
            lower_reward.append(np.min(rewards[-greedy_episode:]))
            avg_greedy_reward.append((np.mean(greedy_reward[-greedy_range:])))

            print("Episode: {}, total numsteps: {}, avg_greedy_reward: {}, average reward: {}".format(
               i_episode, total_numsteps, avg_greedy_reward[-1], np.mean(rewards[-greedy_episode:])))


    #-- saves model --greedy_episode
    if args.save_agent:
        agent.save_model(args.env_name, args.batch_size, '.pth')

    print('Training ended after {} minutes'.format((time.time() - t_start)/60))
    print('Time per ep : {} s').format((time.time() - t_start) / args.num_episodes)
    print('Mean greedy reward: {}'.format(np.mean(greedy_reward)))

    # -- plot learning curve --
    pos_greedy = []
    for pos in range(0, len(lower_reward)):
        pos_greedy.append(pos*greedy_episode)

    plt.fill_between(pos_greedy, lower_reward, upper_reward, facecolor='red', alpha=0.3)
    plt.plot(pos_greedy, avg_greedy_reward, 'r')
    print('Mean reward: {}'.format(np.mean(rewards)))
    print('Max reward: {}'.format(np.max(rewards)))
    print('Min reward: {}'.format(np.min(rewards)))
    plt.plot(rewards, 'r')
    fname = 'plot_{}_{}_{}'.format(args.env_name, args.batch_size, '.png')
    plt.savefig(fname)


if __name__ == '__main__':
    main()

