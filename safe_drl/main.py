import argparse
import os
import math
from collections import namedtuple
from itertools import count
from tqdm import tqdm
from tensorboardX import SummaryWriter

import time
import gym
import numpy as np
from gym import wrappers

import torch
import matplotlib.pyplot as plt

#import files...
from naf import NAF
from ounoise import OUNoise
from replay_memory import Transition, ReplayMemory
from normalized_actions import NormalizedActions

t_start = time.time()

parser = argparse.ArgumentParser(description='PyTorch X-job')
parser.add_argument('--env_name', default="Pendulum-v0",
                    help='name of the environment')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001,
                    help='discount factor for model (default: 0.001)')
parser.add_argument('--ou_noise', type=bool, default=True)
parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                    help='initial noise scale (default: 0.3)')
parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                    help='final noise scale (default: 0.3)')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=4, metavar='N',
                    help='random seed (default: 4)')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--num_steps', type=int, default=10000, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=1000, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--save_agent', type=bool, default=True,
                    help='save model to file')
parser.add_argument('--load_agent', type=bool, default=False,
                    help='load model from file')

args = parser.parse_args()

#env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
writer = SummaryWriter()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


# -- initialize agent, Q and Q' --
agent = NAF(args.gamma, args.tau, args.hidden_size,
            env.observation_space.shape[0], env.action_space, device)
torch.cuda.init()

# -- load existing model --
if args.load_agent and os.path.exists('models/'):
    model_path = "models/naf_{}_{}".format(args.env_name, '.pth')
    agent.load_model(model_path)

# -- declare memory buffer and random process N --
memory = ReplayMemory(args.replay_size)
ounoise = OUNoise(env.action_space.shape[0]) if args.ou_noise else None

rewards = []
total_numsteps = 0
updates = 0
greedy_reward = []
avg_greedy_rewards = []
upper_reward = []
lower_reward = []

for i_episode in range(args.num_episodes):
    # -- reset environment for every episode --
    state = torch.Tensor([env.reset()]).to(device)

    # -- initialize noise (random process N) --
    if args.ou_noise:
        ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                          i_episode / args.exploration_end + args.final_noise_scale)
        ounoise.reset()

    episode_reward = 0

    while(True):
        # -- action selection, observation and store transition --
        torch.cuda.synchronize()
        action = agent.select_action(state, ounoise)
        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
        total_numsteps += 1
        episode_reward += reward
        mask = torch.cuda.FloatTensor([not done])
        next_state = torch.cuda.FloatTensor([next_state])
        reward = torch.cuda.FloatTensor([reward])
        memory.push(state, action, mask, next_state, reward)
        state = next_state

        # -- training --
        if len(memory) > args.batch_size:
            torch.cuda.synchronize()
            for _ in range(args.updates_per_step):
                transitions = memory.sample(args.batch_size)
                batch = Transition(*zip(*transitions))

                value_loss, policy_loss = agent.update_parameters(batch)

                writer.add_scalar('loss/value', value_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)

                updates += 1

        if done:
            break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    rewards.append(episode_reward)

    # -- calculates every 10th episode without N --
    if i_episode % 10 == 0:
        for _ in range(0,9):
            state = torch.cuda.FloatTensor([env.reset()])
            episode_reward = 0
            while True:
                #env.render()
                torch.cuda.synchronize()
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
                episode_reward += reward

                next_state = torch.cuda.FloatTensor([next_state])
                state = next_state
                if done:

                    greedy_reward.append(episode_reward)
                    break

        writer.add_scalar('reward/test', episode_reward, i_episode)

        upper_reward.append(np.max(rewards[-10:]))
        lower_reward.append(np.min(rewards[-10:]))
        avg_greedy_rewards.append(np.mean(greedy_reward[-10:]))

        print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode, total_numsteps, episode_reward, np.mean(rewards[-10:])))

# -- saves model --
if args.save_agent:
    agent.save_model(args.env_name, '.pth')

print("Number of training episodes: {}, done").format(args.num_episodes)
print('Total time : {} m').format((time.time() - t_start) / 60)
print('Time per episode : {} s').format((time.time() - t_start) / args.num_episodes)

pos_greedy = []
for j in range(0, len(avg_greedy_rewards)):
    pos_greedy.append(j*10)

plt.fill_between(pos_greedy, lower_reward, upper_reward, facecolor='red', alpha=0.3)
plt.plot(pos_greedy, avg_greedy_rewards, 'r')
plt.show()

env.close()
