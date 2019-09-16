from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset

import random, json, ast, os, copy, sys
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
import numpy as np
import pandas as pd

import time
from sklearn.manifold import TSNE
import seaborn as sns

from math import pi as PI

from collections import Counter

from nn_models.model_trajectory_vae import VAE as VAE

import rospkg
rospack = rospkg.RosPack()
package_path = rospack.get_path("trajectory_generator")

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='learning loop')
parser.add_argument('--batch-size', type=int, default=12, metavar='N',
                    help='input batch size for training (default: 12)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--debug', nargs='?', const=True, default=False,
                    help='debug flag')
parser.add_argument('--save-dir', default=package_path + "/saved_models/learning_loop/",
                    help='directory where to save the model once trained')
parser.add_argument('--save-file', default=False,
                    help='name of the file to save the model once trained')
parser.add_argument('--load-dir', default=package_path + "/saved_models/learning_loop/",
                    help='directory from where to load the model')
parser.add_argument('--decoder-dir', default=package_path + "/saved_models/trajectory_vae/",
                    help='directory from where to load the model of the action decoder')
parser.add_argument('--decoder-file', default=False,
                    help='file from where to load the model of the action decoder')
parser.add_argument('--load-file', default=False,
                    help='name of the file to load the model from')
parser.add_argument('--learning-rate', type=float, default=0.01,
                    help='value of the learning rate')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='value of the discount factor')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

if not args.decoder_file:
    print ("No decoder specified: provide the file name of the decoder model using the '--decoder-file' argument")
    sys.exit(2)
else:
    model_sd = torch.load(args.decoder_dir+args.decoder_file)
    decoder_in_dim = len(model_sd["fc21.bias"])
    decoder_model = VAE(decoder_in_dim).to(device)
    decoder_model.load_state_dict(model_sd)
    decoder_model.eval()

class Policy(nn.Module):
    def __init__(self, encoded_perception_dimensions):
        super(Policy, self).__init__()
        self.state_space = encoded_perception_dimensions
        self.action_space = decoder_in_dim
        
        # self.l1 = nn.Linear(self.state_space, 128, bias=False)
        # self.l2 = nn.Linear(128, self.action_space, bias=False)

        self.fc1 = nn.Linear(self.state_space, 24)
        
        self.fc21 = nn.Linear(24, 24)  # mean layer
        self.fc31 = nn.Linear(24, self.action_space)

        self.fc22 = nn.Linear(24, 24)  # logvar layer
        self.fc32 = nn.Linear(24, self.action_space)
        
        self.gamma = args.gamma
        
        # Episode policy and reward history 
        self.batch_policies = torch.Tensor()
        self.batch_rewards = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        
        h21 = self.fc21(h)
        mean = self.fc31(h21)

        h22 = self.fc22(h)
        logvar = self.fc32(h22)

        return mean, logvar

    def reparameterize(self, mean, logvar, no_noise):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        if no_noise:
            return mean
            # return mean + std
        
        return mean + eps*std

    def forward(self, x, no_noise):
        mean, logvar = self.encode(x)
        action_sample = self.reparametrize(mean, logvar, no_noise)

        return action_sample

        # model = torch.nn.Sequential(
        #     self.fc1,
        #     nn.Dropout(p=0.6),
        #     nn.ReLU(),
        #     self.l2,
        #     nn.Softmax(dim=-1)
        # )
        # return model(x)

def update_policy():
    R = 0
    rewards = []
    
    # Discount future rewards back to the present using gamma
    # for r in policy_model.batch_rewards[::-1]:
    #     R = r + policy_model.gamma * R
    #     rewards.insert(0,R)
        
    # Scale rewards
    # rewards = torch.FloatTensor(rewards)
    # rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    
    # Calculate loss
    loss = (torch.sum(torch.mul(policy_model.batch_policies, policy_model.batch_rewards).mul(-1), -1))
    
    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #Save and intialize episode history counters
    policy_model.loss_history.append(loss.data[0])
    policy_model.reward_history.append(np.sum(policy_model.batch_rewards))
    policy_model.batch_policies = torch.Tensor()
    policy_model.batch_rewards= []

policy_model = Policy(1)
optimizer = optim.Adam(policy_model.parameters(), lr=args.learning_rate)