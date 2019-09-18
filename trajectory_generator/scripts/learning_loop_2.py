from __future__ import print_function
import argparse
import torch
import math
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

import importlib

from torch.distributions.multivariate_normal import MultivariateNormal

# from nn_models.model_trajectory_vae import VAE as VAE
class VAE(nn.Module):
    def __init__(self, latent_space_dim):
        super(VAE, self).__init__()
        
        self.fc1 = nn.Linear(700, 400)
        self.fc21 = nn.Linear(400, latent_space_dim)  # mu layer
        self.fc22 = nn.Linear(400, latent_space_dim)  # logvariance layer
        self.fc3 = nn.Linear(latent_space_dim, 400)
        self.fc4 = nn.Linear(400, 700)
        self.fc5 = nn.Linear(700, 700)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar, no_noise):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        if no_noise:
            return mu
            # return mu + std
        
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = torch.sigmoid(self.fc4(h3))
        # return h4
        return self.fc5(h4)

    def forward(self, x, no_noise):
        mu, logvar = self.encode(x.view(-1, 700))
        z = self.reparameterize(mu, logvar, no_noise)
        return self.decode(z), mu, logvar

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
parser.add_argument('--load-file', default=False,
                    help='name of the file to load the model from')
parser.add_argument('--decoder-dir', default=package_path + "/saved_models/trajectory_vae/",
                    help='directory from where to load the trained model of the action decoder')
parser.add_argument('--decoder-file', default=False,
                    help='file from where to load the trained model of the action decoder')
parser.add_argument('--save-policy-dir', default=package_path + "/saved_models/policy_network/",
                    help='directory where to save the trained model of the policy network')
parser.add_argument('--save-policy', default=False,
                    help='file where to save the trained model of the policy network')
parser.add_argument('--load-policy-dir', default=package_path + "/saved_models/policy_network/",
                    help='directory where to load the trained model of the policy network')
parser.add_argument('--load-policy', default=False,
                    help='file where to load the trained model of the policy network')
parser.add_argument('--policy-model-dir', default="nn_models",
                    help='directory where to load the trained model of the policy network')
parser.add_argument('--policy-model', default=False,
                    help='file where to load the trained model of the policy network')
parser.add_argument('--state-dir', default=package_path + "/saved_models/trajectory_vae/",
                    help='directory from where to load the trained model of the state encoder')
parser.add_argument('--state-file', default=False,
                    help='file from where to load the trained model of the state encoder')
parser.add_argument('--algorithm-dir', default="learning_algorithms",
                    help='directory from where to load the learning algorithm')
parser.add_argument('--algorithm', default=False,
                    help='file from where to load the learning algorithm')
parser.add_argument('--learning-rate', type=float, default=0.01,
                    help='value of the learning rate')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='value of the discount factor')
parser.add_argument('--no-noise', nargs='?', const=True, default=False,
                    help='flag for the added noise in action sampling')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

initial_state = torch.zeros(1)

pi = torch.FloatTensor([math.pi])

class Policy(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        super(Policy, self).__init__()
        self.state_space = state_space_dim
        self.action_space = action_space_dim
        
        # self.l1 = nn.Linear(self.state_space, 128, bias=False)
        # self.l2 = nn.Linear(128, self.action_space, bias=False)

        self.fc1 = nn.Linear(self.state_space, 24)
        
        self.fc21 = nn.Linear(24, 24)  # mean layer
        self.fc31 = nn.Linear(24, self.action_space)

        self.fc22 = nn.Linear(24, 24)  # log_var layer
        self.fc32 = nn.Linear(24, self.action_space)

        # self.p1 = torch.Variable(torch.randn(1))
        # self.p2 = torch.Variable(torch.randn(1))

        self.p1 = torch.Tensor(torch.randn(1))
        self.p2 = torch.Tensor(torch.randn(1))

        self.p1 = torch.nn.Parameter(torch.randn(2))
        # self.p2 = torch.nn.Parameter(torch.randn(1))

        # self.p1 = nn.Linear(1, 1)
        # self.p2 = nn.Linear(1, 1)

    
    def normal_prob(self, x, mean, sigma_sq):
        a = (-1*(x-mean).pow(2)/(2*sigma_sq)).exp()
        b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
        return a*b
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        
        h21 = self.fc21(h)
        mean = self.fc31(h21)

        h22 = self.fc22(h)
        log_var = self.fc32(h22)

        return mean, log_var

    def sample_action(self, mean, log_var, no_noise):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        
        if no_noise:
            return mean
            # return mean + std
        
        return mean + eps*std

    def forward(self, x, no_noise):
        # mean = torch.Tensor([self.p1, self.p2])
        mean = self.p1
        log_var = torch.randn(2)
        
        sigma_sq = torch.tensor([0.01, 0.01])
        # mean, log_var = self.encode(x)
        # sigma_sq = torch.exp(log_var)

        # =============== NEW CODE ===============
        # cov_matrix = torch.diag(sigma_sq)
        # cov_matrix = torch.diag(torch.FloatTensor([0.001, 0.001, 0.001, 0.001, 0.001]))
        cov_matrix = torch.diag(torch.FloatTensor([0.001, 0.001]))
        dist = MultivariateNormal(mean, cov_matrix)
        action_sample = dist.sample()
        print (mean)
        log_prob = dist.log_prob(action_sample)
        # =============== NEW CODE ===============
        
        # =============== OLD CODE ===============
        # action_sample = self.sample_action(mean, log_var, no_noise).data
        # action_prob = self.normal_prob(action_sample, mean, sigma_sq)
        # log_prob = action_prob.log()
        # =============== OLD CODE ===============
        entropy = -0.5*((sigma_sq+2*pi.expand_as(sigma_sq)).log()+1)

        # return action_sample, mean, log_var, log_prob, entropy
        return mean, mean, log_var, mean, entropy

        # model = torch.nn.Sequential(
        #     self.fc1,
        #     nn.Dropout(p=0.6),
        #     nn.ReLU(),
        #     self.l2,
        #     nn.Softmax(dim=-1)
        # )
        # return model(x)

if not args.decoder_file:
    print ("No decoder specified: provide the file name of the decoder model using the '--decoder-file' argument")
    sys.exit(2)
else:
    model_sd = torch.load(args.decoder_dir+args.decoder_file)
    decoder_in_dim = len(model_sd["fc21.bias"])
    decoder_model = VAE(decoder_in_dim).to(device)
    decoder_model.load_state_dict(model_sd)
    decoder_model.eval()

if not args.policy_model:
    print ("No policy model specified: provide the file name of the policy model using the '--policy-model' argument")
    sys.exit(2)
else:

    # RIGHT ===========================
    # policy_module = importlib.import_module(args.policy_model_dir + "." + args.policy_model)
    # # policy_model = policy_module.Policy(initial_state.dim(), decoder_in_dim)
    # policy_model = policy_module.Policy(initial_state.dim(), 2)
    # TEST ===========================
    policy_model = Policy(initial_state.dim(), 2)
    # ===========================

    optimizer = optim.Adam(policy_model.parameters(), lr=args.learning_rate)

# Implementation of the reinforce algorithm for a continuous action space
class ALGORITHM:
    def __init__(self, policy_model, optimizer, gamma):
        self.policy_model = policy_model
        self.optimizer = optimizer
        self.gamma = gamma

    def update_policy(self, log_probs, rewards, entropies):
        R = 0
        loss = 0

        for i in reversed(range(len(rewards))):
            # R = gamma * R + rewards[i]
            loss = loss + (log_probs[i]*rewards[i]).sum()
        loss = loss / len(rewards)

        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        # utils.clip_grad_norm(self.policy_model.parameters(), 40)
        self.optimizer.step()
        return loss

if not args.algorithm:
    print ("No learning algorithm specified: provide the file name of the learning algorithm using the '--algorithm' argument")
    sys.exit(2)
else:
    # algorithm_module = importlib.import_module(args.algorithm_dir + "." + args.algorithm)
    # algorithm = algorithm_module.ALGORITHM(policy_model, optimizer, args.gamma)
    algorithm = ALGORITHM(policy_model, optimizer, args.gamma)

def update_graph(fig, ax, line, x_value, y_value):
    
    line.set_xdata(np.append(line.get_xdata(), x_value))
    line.set_ydata(np.append(line.get_ydata(), y_value))
    # plt.draw()

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

live_plots = {
    "loss": {
        "fig": None,
        "ax": None,
        "line1": None
    },
    "reward": {
        "fig": None,
        "ax": None,
        "line1": None
    }
}
plt.ion()
live_plots["loss"]["fig"] = plt.figure("Loss")
# live_plots["loss"]["fig"].suptitle('Loss', fontsize=10)
live_plots["loss"]["ax"] = live_plots["loss"]["fig"].add_subplot(2, 1, 1)
# live_plots["loss"]["ax"].set_title("Loss")
live_plots["loss"]["line1"], = live_plots["loss"]["ax"].plot([], [], 'o-r', label="TRAIN Loss")
live_plots["loss"]["zero"], = live_plots["loss"]["ax"].plot([], [], '-k')

live_plots["reward"]["fig"] = live_plots["loss"]["fig"]
live_plots["reward"]["ax"] = live_plots["loss"]["fig"].add_subplot(2, 1, 2)
# live_plots["reward"]["ax"].set_title("Reward")
live_plots["reward"]["line1"], = live_plots["reward"]["ax"].plot([], [], 'o-b', label="Reward")
live_plots["reward"]["zero"], = live_plots["reward"]["ax"].plot([], [], '-k')

# live_plots["kld_loss"]["fig"] = live_plots["loss"]["fig"]
# live_plots["kld_loss"]["ax"] = live_plots["loss"]["ax"]
# live_plots["kld_loss"]["line1"], = live_plots["loss"]["ax"].plot([], [], '-k', label="KLD Loss")

# live_plots["test_loss"]["fig"] = live_plots["loss"]["fig"]
# live_plots["test_loss"]["ax"] = live_plots["loss"]["ax"]
# live_plots["test_loss"]["line1"], = live_plots["loss"]["ax"].plot([], [], 'x-b', label="TEST Loss")

live_plots["loss"]["ax"].legend()
live_plots["reward"]["ax"].legend()

history = []
# history item:
# {
#     "state": value
#     "action": value
#     "mean": value
#     "log_var": value
#     "log_prob": value
#     "entropy": value
#     "reward": value
# }

# loss of each batch
loss_history = []

def execute_action(action):
    error = False
    reward = ((-1)*abs(action - 2)).sum().item()
    # reward = ((-1)*(action - 2)**2).sum().item()
    # reward = reward / action.dim()
    return error, reward

def train():
    for epoch in range(args.epochs):
        print ("Epoch ", str(epoch+1))
        entropies = []
        log_probs = []
        rewards = []
        for batch_index in range(args.batch_size):

            # action, mean, log_var, log_prob, entropy = policy_model(initial_state, args.no_noise)
            action, mean, log_var, log_prob, entropy = policy_model(initial_state, args.no_noise)
            error, reward = execute_action(action)
            # print ("\n\tAttempt ", str(batch_index+1))
            # print ("\taction = ", str(action))
            # print ("\treward = ", str(reward))

            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)
            history.append(
                {
                    "state": initial_state,
                    "action": action,
                    "mean": mean,
                    "log_var": log_var,
                    "log_prob": log_prob,
                    "entropy": entropy,
                    "reward": reward
                }
            )

        loss = algorithm.update_policy(log_probs, rewards, entropies)
        loss_history.append(loss)
        # print ("loss = ", str(loss))
        update_graph(live_plots["loss"]["fig"], live_plots["loss"]["ax"], live_plots["loss"]["line1"], epoch, loss.item())
        update_graph(live_plots["reward"]["fig"], live_plots["reward"]["ax"], live_plots["reward"]["line1"], epoch, reward)
        True
        


def main(args):
    
    print ("Start training")
    train()
    print ("Done training")

    plt.show()

if __name__ == '__main__':
    main(sys.argv)  