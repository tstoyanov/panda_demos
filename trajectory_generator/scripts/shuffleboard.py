from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset

import random, json, ast, os, copy
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

# debug purpose?
# import multiprocessing
# multiprocessing.set_start_method('spawn', True)

import rospkg
rospack = rospkg.RosPack()
package_path = rospack.get_path("trajectory_generator")

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser(description='VAE Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--dataset-dir', default="latest_batch",
                    help='path of the directory containing the input dataset')
parser.add_argument('--debug', nargs='?', const=True, default=False,
                    help='debug flag')
parser.add_argument('--save-dir', default=package_path + "/saved_model/",
                    help='directory where to save the model once trained')
parser.add_argument('--save-file', default=False,
                    help='name of the file to save the model once trained')
parser.add_argument('--load-dir', default=package_path + "/saved_model/",
                    help='directory from where to load the model')
parser.add_argument('--load-file', default=False,
                    help='name of the file to load the model from')
parser.add_argument('--tsne', nargs='?', const=True, default=False,
                    help='whether or not to perform and plot t-sne on the embedded data')
parser.add_argument('--loss-type', default="MSE_SUM",
                    help='type of loss metric to combine with KLD while calculating the loss')
parser.add_argument('--alpha', default=1,
                    help='value of the alpha parameter used recostruction_loss weight in loss calculation')
parser.add_argument('--beta', default=1,
                    help='value of the beta parameter used as KLD weight in loss calculation')
parser.add_argument('--norm', nargs='?', const=True, default=False,
                    help='whether to apply normalization or not')
parser.add_argument('--test-percentage', default=10,
                    help='percentage of the dataset to use as test data (0, 100)')
parser.add_argument('--validation-percentage', default=10,
                    help='percentage of the dataset to use as validation data (0, 100)')
parser.add_argument('--random', nargs='?', const=True, default=False,
                    help='whether to randomize the test set initialization or not')
parser.add_argument('--joints-plot', nargs='?', const=True, default=False,
                    help='whether to plot the decoded joints position vs the original ones or not')
parser.add_argument('--transformation-plot', nargs='?', const=10, default=False,
                    help='number of steps for the transformation from one embedded trajetory to another one')
parser.add_argument('--matrix-plot', nargs='?', const=True, default=False,
                    help='whether to plot the intersection between pairs of dimensions in the latent space or not')
parser.add_argument('--deg', nargs='?', const=True, default=False,
                    help='whether to transform radians into degrees or not')
parser.add_argument('--wmb', nargs='?', const=True, default=False,
                    help='whether to show the trajectories with the worst, median and best loss or not')
parser.add_argument('--write', nargs='?', const=True, default=False,
                    help='whether to write the generated trajectory to the rostopic or not')


args = parser.parse_args()
if args.dataset_dir[0] != "/":
    args.dataset_dir = "/" + args.dataset_dir
if args.dataset_dir[-1] != "/":
    args.dataset_dir = args.dataset_dir + "/"
args.dataset_dir = package_path + "/generated_trajectories/datasets" + args.dataset_dir
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if args.deg:
    coeff = 180/PI
else:
    coeff = 1

if not args.save_dir:
    args.save_dir = package_path + "/saved_model/"
if not args.load_dir:
    args.load_dir = package_path + "/saved_model/"

get_encoded_data = args.tsne != False or args.matrix_plot != False

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)

latent_space_train = None
latent_space_test = None

joints_ranges = {
    "panda_joint1": {
        "min": -2.8973,
        "max": 2.8973
    },
    "panda_joint2": {
        "min": -1.7628,
        "max": 1.7628
    },
    "panda_joint3": {
        "min": -2.8973,
        "max": 2.8973
    },
    "panda_joint4": {
        "min": -3.0718,
        "max": -0.0698
    },
    "panda_joint5": {
        "min": -2.8973,
        "max": 2.8973
    },
    "panda_joint6": {
        "min": -0.0175,
        "max": 3.7525
    },
    "panda_joint7": {
        "min": -2.8973,
        "max": 2.8973
    }
}

def shuffle_lists(*args):
    for i in reversed(range(1, len(args[0]))):
        j = int(random.random() * (i+1))
        for x in args: 
            x[i], x[j] = x[j], x[i]

def update_graph(fig, ax, line1, x_value, y_value):
    
    line1.set_xdata(np.append(line1.get_xdata(), x_value))
    line1.set_ydata(np.append(line1.get_ydata(), y_value))
    # plt.draw()

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


def data_split(input_dataset, test_percentage, validation_percentage):
    input_train_set = input_dataset + "train_set.txt"
    input_test_set = input_dataset + "test_set.txt"
    input_validation_set = input_dataset + "validation_set.txt"
    input_dataset += "dataset.txt"
    assert os.path.isfile(input_dataset)

    if os.path.isfile(input_train_set) and os.path.isfile(input_test_set) and os.path.isfile(input_validation_set):
        print ("opening train_set...")
        with open(input_train_set, 'r') as f:
            print ("loading train_set...")
            train_set = json.loads(f.read())
        print ("train_set loaded")

        print ("opening test_set...")
        with open(input_test_set, 'r') as f:
            print ("loading test_set...")
            test_set = json.loads(f.read())
        print ("test_set loaded")
        
        print ("opening validation_set...")
        with open(input_validation_set, 'r') as f:
            print ("loading validation_set...")
            validation_set = json.loads(f.read())
        print ("validation_set loaded")

    else:
        print ("opening dataset...")
        with open(input_dataset, 'r') as f:
            print ("loading dataset...")
            train_set = json.loads(f.read())
        print ("dataset loaded")

        print ("generating train_set, test_set and validation_set...")
        dataset_length = len(train_set["joints_positions"])
        # train_set["eef_velocity_magnitude"] = list(map(lambda vel: round(vel, 1) , train_set["eef_velocity_magnitude"]))
        labels = set(train_set["eef_velocity_magnitude"])
        test_set = {
            "joints_positions": [],
            "eef_velocity_magnitude": [],
            "m": [],
            "c": []
        }
        validation_set = {
            "joints_positions": [],
            "eef_velocity_magnitude": [],
            "m": [],
            "c": []
        }

        # random test set initialization
        if args.random == True:
            for i in range(int(dataset_length * float(test_percentage) / 100)):
                sample_index = random.randint(0,len(train_set["joints_positions"])-1)
                test_set["joints_positions"].append(train_set["joints_positions"].pop(sample_index))
                test_set["eef_velocity_magnitude"].append(train_set["eef_velocity_magnitude"].pop(sample_index))
                test_set["m"].append(train_set["m"].pop(sample_index))
                test_set["c"].append(train_set["c"].pop(sample_index))
            for i in range(int(dataset_length * float(validation_percentage) / 100)):
                sample_index = random.randint(0,len(train_set["joints_positions"])-1)
                validation_set["joints_positions"].append(train_set["joints_positions"].pop(sample_index))
                validation_set["eef_velocity_magnitude"].append(train_set["eef_velocity_magnitude"].pop(sample_index))
                validation_set["m"].append(train_set["m"].pop(sample_index))
                validation_set["c"].append(train_set["c"].pop(sample_index))

        # non-random test set initialization
        else:
            shuffle_lists(train_set["eef_velocity_magnitude"], train_set["joints_positions"])

            subsets_composition = list(map(lambda item: ((item[0], int(item[1]*test_percentage/100)), (item[0], int(item[1]*validation_percentage/100))), list(Counter(train_set["eef_velocity_magnitude"]).items())))
            test_composition = [item[0] for item in subsets_composition]
            validation_composition = [item[1] for item in subsets_composition]
            
            for (label, amount) in test_composition:
                for i in range(amount):
                    try:
                        sample_index = train_set["eef_velocity_magnitude"].index(label)
                        test_set["joints_positions"].append(train_set["joints_positions"].pop(sample_index))
                        test_set["eef_velocity_magnitude"].append(train_set["eef_velocity_magnitude"].pop(sample_index))
                        test_set["m"].append(train_set["m"].pop(sample_index))
                        test_set["c"].append(train_set["c"].pop(sample_index))
                    except:
                        break
            while len(test_set["joints_positions"]) < dataset_length * test_percentage / 100:
                sample_index = random.randint(0,len(train_set["joints_positions"])-1)
                test_set["joints_positions"].append(train_set["joints_positions"].pop(sample_index))
                test_set["eef_velocity_magnitude"].append(train_set["eef_velocity_magnitude"].pop(sample_index))
                test_set["m"].append(train_set["m"].pop(sample_index))
                test_set["c"].append(train_set["c"].pop(sample_index))
            
            for (label, amount) in validation_composition:
                for i in range(amount):
                    try:
                        sample_index = train_set["eef_velocity_magnitude"].index(label)
                        validation_set["joints_positions"].append(train_set["joints_positions"].pop(sample_index))
                        validation_set["eef_velocity_magnitude"].append(train_set["eef_velocity_magnitude"].pop(sample_index))
                        validation_set["m"].append(train_set["m"].pop(sample_index))
                        validation_set["c"].append(train_set["c"].pop(sample_index))
                    except:
                        break
            while len(validation_set["joints_positions"]) < dataset_length * validation_percentage / 100:
                sample_index = random.randint(0,len(train_set["joints_positions"])-1)
                validation_set["joints_positions"].append(train_set["joints_positions"].pop(sample_index))
                validation_set["eef_velocity_magnitude"].append(train_set["eef_velocity_magnitude"].pop(sample_index))
                validation_set["m"].append(train_set["m"].pop(sample_index))
                validation_set["c"].append(train_set["c"].pop(sample_index))
        
        print ("train_set, test_set and validation_set generated")
        print ("saving train_set, test_set and validation_set...")
        # python 2
        # os.makedirs(os.path.dirname(input_train_set))
        # python 3
        os.makedirs(os.path.dirname(input_train_set), exist_ok=True)
        with open(input_train_set, "w") as f:
            json.dump(train_set, f)
        # python 2
        # os.makedirs(os.path.dirname(input_test_set))
        # python 3
        os.makedirs(os.path.dirname(input_test_set), exist_ok=True)
        with open(input_test_set, "w") as f:
            json.dump(test_set, f)
        # python 2
        # os.makedirs(os.path.dirname(input_validation_set))
        # python 3
        os.makedirs(os.path.dirname(input_train_set), exist_ok=True)
        with open(input_validation_set, "w") as f:
            json.dump(validation_set, f)
        print ("train_set, test_set and validation_set saved")
    
    return train_set, test_set, validation_set


def joints_plot (last_data, last_recon_data, title):
    joints_ranges_list = list(joints_ranges.items())
    fig = plt.figure()
    if title == None:
        fig.suptitle('Decoded vs Original', fontsize=10)
    else:
        fig.suptitle(title, fontsize=10)

    for joint_index, _ in enumerate(last_data[0]):
        # ax = fig.add_subplot(4, 2, joint_index+1)
        ax = fig.add_subplot(len(last_data[0]), 1, joint_index+1)
        ax.set_title("joint_" + str(joint_index+1))
        if "Absolute difference between reconstructed and original trajectory from test set" == title or "Absolute difference between reconstructed and original trajectory from train set" == title:
            ax.plot(range(len(last_data)), [item[joint_index].item()*coeff for item in last_data], 'o-b', label="noise")
            ax.plot(range(len(last_recon_data)), [item[joint_index].item()*coeff for item in last_recon_data], 'x-r', label="no noise")
            ax.legend()
        elif "transformation_plot" == title:
            fig.suptitle("Iteration: "+str(last_recon_data), fontsize=10)
            ax.plot(range(len(last_data)), [item[joint_index].item()*coeff for item in last_data], 'o-b')
        else:
            ax.plot(range(len(last_data)), [item[joint_index].item()*coeff for item in last_data], 'o-b', label="Original")
            ax.plot(range(len(last_recon_data)), [item[joint_index].item()*coeff for item in last_recon_data], 'x-r', label="Reconstructed")
            ax.legend()
            # ax.set(ylim=(joints_ranges_list[joint_index][1]["min"], joints_ranges_list[joint_index][1]["max"]))

def new_joints_plot (last_data, last_recon_data, title, fig, color):
    joints_ranges_list = list(joints_ranges.items())
    # fig = plt.figure()
    if title == None:
        fig.suptitle('Decoded vs Original', fontsize=10)
    else:
        fig.suptitle(title, fontsize=10)

    new_index = -1
    for joint_index, _ in enumerate(last_data[0]):
        # 3 4 6
        joints_to_plot = [2, 3, 5]
        if joint_index in joints_to_plot:
            new_index += 1
            # ax = fig.add_subplot(4, 2, joint_index+1)
            ax = fig.add_subplot(len(joints_to_plot), 1, new_index+1)
            # ax.set_title("joint_" + str(joint_index+1))
            fig.suptitle("Iteration: "+str(last_recon_data), fontsize=10)
            ax.plot(range(len(last_data)), [item[joint_index].item()*coeff for item in last_data], '-', marker='o', markersize=5, color=color, linewidth=1, alpha=0.7)
            ax.set_ylabel('joint angle [rad]')
            if new_index+1 == len(joints_to_plot):
                ax.set_xlabel('steps')

class MyDataset(Dataset):
    """
    This dataset contains the trajectories stored in the file passed as argument
    The __getitem__ method returns a whole trajectory
    """
    def __init__(self, input_dataset):
        super(MyDataset, self).__init__()
        
        self.dataset = input_dataset

        # assert os.path.isfile(input_dataset)
        # self.input_dataset = input_dataset
        # print ("opening dataset...")
        # with open(input_dataset, 'r') as f:
        #     print ("loading dataset...")
        #     self.dataset = json.loads(f.read())
        # 
        # self.flat_dataset = []
        # for i in range(len(self.dataset)):
        #     self.flat_dataset.append([joint_value for joints_values in self.dataset[i] for joint_value in joints_values])
        
    def __len__(self):
        return len(self.dataset["joints_positions"])
        # return len(self.flat_dataset)
        
    def __getitem__(self, index):
        assert 0 <= index < len(self.dataset["joints_positions"])
        # assert 0 <= index < len(self.flat_dataset)
        
        # joints range regularization
        if args.norm != False:
            ret = []
            for joint_index, joints_positions in enumerate(self.dataset["joints_positions"][index]):
                ret.append(list(map(lambda joint_value, joint_name: (joint_value-joints_ranges[joint_name]["min"])/(joints_ranges[joint_name]["max"]-joints_ranges[joint_name]["min"]), joints_positions, joints_ranges)))
            return torch.tensor(ret), torch.tensor(round(self.dataset["eef_velocity_magnitude"][index], 1)), torch.tensor(round(self.dataset["m"][index], 4)), torch.tensor(round(self.dataset["c"][index], 4)), index
        else:
            discretized_m = (self.dataset["m"][index]*100 // 0.3) * 0.3
            discretized_c = (self.dataset["c"][index]*100 // 0.3) * 0.3
            return torch.tensor(self.dataset["joints_positions"][index]), torch.tensor(round(self.dataset["eef_velocity_magnitude"][index], 1)), torch.tensor(discretized_m), torch.tensor(discretized_c), index
            # return torch.tensor(self.dataset["joints_positions"][index]), torch.tensor(round(self.dataset["eef_velocity_magnitude"][index], 1)), torch.tensor(round(self.dataset["m"][index]*100, 1)), torch.tensor(round(self.dataset["c"][index]*100, 1)), index

        # return torch.tensor(self.flat_dataset[index]), index
        # return self.flat_dataset[index], index
        # return self.dataset[index], index

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.fc1 = nn.Linear(700, 400)
        self.fc21 = nn.Linear(400, 5)  # mu layer
        self.fc22 = nn.Linear(400, 5)  # logvariance layer
        self.fc3 = nn.Linear(5, 400)
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


learning_rate = 0.01
gamma = 0.99
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.encoded_perception_dimensions = 1
        self.encoded_action_dimensions = 5
        
        # self.l1 = nn.Linear(self.state_space, 128, bias=False)
        # self.l2 = nn.Linear(128, self.action_space, bias=False)

        self.fc1 = nn.Linear(encoded_perception_dimensions, 24)
        
        self.fc21 = nn.Linear(24, 24)  # mean layer
        self.fc31 = nn.Linear(24, 5)

        self.fc22 = nn.Linear(24, 24)  # std layer
        self.fc32 = nn.Linear(24, 5)
        
        self.gamma = gamma
        
        # Episode policy and reward history 
        self.policy_history = torch.Tensor()
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        
        h21 = self.fc21(h)
        mean = self.fc31(h21)

        h22 = self.fc22(h)
        std = self.fc32(h22)

        return mean, std

    def reparameterize(self, mean, std, no_noise):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        if no_noise:
            return mu
            # return mu + std
        
        return mu + eps*std

    def forward(self, x, no_noise):
        mean, std = self.encode(x)
        action_sample = self.reparametrize(mean, std, no_noise)

        return action_sample

        # model = torch.nn.Sequential(
        #     self.fc1,
        #     nn.Dropout(p=0.6),
        #     nn.ReLU(),
        #     self.l2,
        #     nn.Softmax(dim=-1)
        # )
        # return model(x)


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, label):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    if "MSE_MEAN" == args.loss_type.upper():
        criterion = nn.MSELoss(reduction='mean')
        reconstruction_loss = criterion(recon_x, x.view(-1, 700))
        return float(args.alpha)*reconstruction_loss + float(args.beta)*KLD, float(args.alpha)*reconstruction_loss, float(args.beta)*KLD
    if "MSE_SUM" == args.loss_type.upper():
        criterion = nn.MSELoss(reduction='sum')
        reconstruction_loss = criterion(recon_x, x.view(-1, 700))
        return float(args.alpha)*reconstruction_loss + float(args.beta)*KLD, float(args.alpha)*reconstruction_loss, float(args.beta)*KLD
    if "MSE_NONE" == args.loss_type.upper():
        criterion = nn.MSELoss(reduction='none')
        reconstruction_loss = criterion(recon_x, x.view(-1, 700))
        return float(args.alpha)*reconstruction_loss + float(args.beta)*KLD, float(args.alpha)*reconstruction_loss, float(args.beta)*KLD

    elif "BCE_SUM" == args.loss_type.upper():
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 700), reduction='sum')
        return float(args.alpha)*BCE + float(args.beta)*KLD
    elif "BCE_MEAN" == args.loss_type.upper():
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 700), reduction='mean')
        return float(args.alpha)*BCE + float(args.beta)*KLD
    elif "BCE_NONE" == args.loss_type.upper():
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 700), reduction='none')
        return float(args.alpha)*BCE + float(args.beta)*KLD

    elif "L1_NORM_MEAN" == args.loss_type.upper() or "MAE_MEAN" == args.loss_type.upper() or "L1_MEAN" == args.loss_type.upper():
        criterion = nn.L1Loss(reduction="mean")
        reconstruction_loss = criterion(recon_x, x.view(-1, 700))
        return float(args.alpha)*reconstruction_loss + float(args.beta)*KLD, float(args.alpha)*reconstruction_loss, float(args.beta)*KLD
    elif "L1_NORM_SUM" == args.loss_type.upper() or "MAE_SUM" == args.loss_type.upper() or "L1_SUM" == args.loss_type.upper():
        criterion = nn.L1Loss(reduction="sum")
        reconstruction_loss = criterion(recon_x, x.view(-1, 700))
        return float(args.alpha)*reconstruction_loss + float(args.beta)*KLD, float(args.alpha)*reconstruction_loss, float(args.beta)*KLD
    elif "L1_NORM_NONE" == args.loss_type.upper() or "MAE_NONE" == args.loss_type.upper() or "L1_NONE" == args.loss_type.upper():
        criterion = nn.L1Loss(reduction="none")
        reconstruction_loss = criterion(recon_x, x.view(-1, 700))
        return float(args.alpha)*reconstruction_loss + float(args.beta)*KLD, float(args.alpha)*reconstruction_loss, float(args.beta)*KLD

    elif "CE_SUM" == args.loss_type.upper():
        CE = F.cross_entropy(recon_x, label.type(torch.LongTensor), reduction='sum')
        return float(args.alpha)*CE + float(args.beta)*KLD
    elif "CE_MEAN" == args.loss_type.upper():
        CE = F.cross_entropy(recon_x, label.type(torch.LongTensor), reduction='mean')
        return float(args.alpha)*CE + float(args.beta)*KLD
    elif "CE_NONE" == args.loss_type.upper():
        CE = F.cross_entropy(recon_x, label.type(torch.LongTensor), reduction='none')
        return float(args.alpha)*CE + float(args.beta)*KLD

    raise ValueError("the value: '" + str(args.loss_type) + "' of the '--loss-type' argument is not valid")


def my_train(epoch, loss_plots):

    model.train()
    # print ("model: ", model)
    train_loss = 0
    for batch_idx, (data, label, m, c, index) in enumerate(my_train_set_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, False)
        # print ("mu.grad_fn: ", mu.grad_fn)
        # print ("mu.grad_fn.next_functions[0][0]: ", mu.grad_fn.next_functions[0][0])
        # print ("logvar.grad_fn: ", logvar.grad_fn)
        loss, recon_loss, kld_loss = loss_function(recon_batch, data, mu, logvar, label)
        # print ("value of model.fc1.bias.grad before backward: ")
        # print (model.fc1.bias.grad)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # print ("model.parameters(): ", model.parameters())
        # for i, f in enumerate(model.parameters()):
        #     print ("i: ", i)
        #     print ("f.data.shape: ", f.data.shape)
            # print ("f.data: ", f.data)
        # raw_input()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(my_train_set_loader.dataset),
                100. * batch_idx / len(my_train_set_loader),
                loss.item() / len(data)))

        if args.batch_size == len(my_train_set) or (batch_idx % 10 == 0 and epoch != 1):
            update_graph(loss_plots["loss"]["fig"], loss_plots["loss"]["ax"], loss_plots["loss"]["line1"], batch_idx + epoch*len(my_train_set_loader), loss.item()/len(data))
        #     update_graph(loss_plots["recon_loss"]["fig"], loss_plots["recon_loss"]["ax"], loss_plots["recon_loss"]["line1"], batch_idx + epoch*len(my_train_set_loader), recon_loss.item()/len(data))
        #     update_graph(loss_plots["kld_loss"]["fig"], loss_plots["kld_loss"]["ax"], loss_plots["kld_loss"]["line1"], batch_idx + epoch*len(my_train_set_loader), kld_loss.item()/len(data))

    # update_graph(loss_plots["loss"]["fig"], loss_plots["loss"]["ax"], loss_plots["loss"]["line1"], len(my_train_set_loader) + epoch*len(my_train_set_loader), loss.item()/len(data))

    if args.joints_plot != False and epoch == args.epochs:
        title = "Random sample from train set with noise\nDecoded vs Original"
        joints_plot(data[0], recon_batch.view(len(recon_batch), 100, 7)[0], title)

        if isinstance(args.joints_plot, str) and ("NO-NOISE" == args.joints_plot.upper() or "NO_NOISE" == args.joints_plot.upper() or "NONOISE" == args.joints_plot.upper()):
            title = "Random sample from train set without noise\nDecoded vs Original"
            recon_data, mu, logvar = model(data[0].unsqueeze(0), True)
            joints_plot(data[0], recon_data.view(1, 100, 7)[0], title)
        # joints_plot(last_data, last_recon_data, title)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(my_train_set_loader.dataset)))


def my_test(epoch, loss_plots):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, label, m, c, index) in enumerate(my_test_set_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data, False)
            test_loss += loss_function(recon_batch, data, mu, logvar, label)[0].item()

            # if batch_idx == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            #     save_image(comparison.cpu(),
            #              'results/reconstruction_' + str(epoch) + '.png', nrow=n)
        
        if args.joints_plot != False and (epoch == args.epochs or args.load_file != False):
            title = "Random sample from test set with noise\nDecoded vs Original"
            joints_plot(data[0], recon_batch.view(len(recon_batch), 100, 7)[0], title)

            if isinstance(args.joints_plot, str) and ("NO-NOISE" == args.joints_plot.upper() or "NO_NOISE" == args.joints_plot.upper() or "NONOISE" == args.joints_plot.upper()):
                title = "Random sample from test set without noise\nDecoded vs Original"
                recon_data, mu, logvar = model(data[0].unsqueeze(0), True)
                joints_plot(data[0], recon_data.view(1, 100, 7)[0], title)

                title = "Absolute difference between reconstructed and original trajectory from test set"
                joints_plot(abs(data[0] - recon_batch.view(len(recon_batch), 100, 7)[0]), abs(data[0] - recon_data.view(1, 100, 7)[0]), title)

    test_loss /= len(my_test_set_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    if loss_plots != None:
        update_graph(loss_plots["test_loss"]["fig"], loss_plots["test_loss"]["ax"], loss_plots["test_loss"]["line1"], len(my_train_set_loader) + epoch*len(my_train_set_loader), test_loss)
    # update_graph(loss_plots["test_loss"]["fig"], loss_plots["test_loss"]["ax"], loss_plots["test_loss"]["line1"], len(my_train_set_loader) + epoch*len(my_train_set_loader), test_loss.item()/len(data))


def save_model_state_dict(save_path):
    torch.save(model.state_dict(), save_path)


def load_model_state_dict(load_path):
    loaded_model = VAE().to(device)
    loaded_model.load_state_dict(torch.load(load_path))
    loaded_model.eval()
    return loaded_model
    # return VAE().to(device).load_state_dict(torch.load(load_path)).eval()

train_set, test_set, validation_set = data_split(args.dataset_dir, args.test_percentage, args.validation_percentage)

my_train_set = MyDataset(train_set)
my_train_set_loader = torch.utils.data.DataLoader(my_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

my_test_set = MyDataset(test_set)
my_test_set_loader = torch.utils.data.DataLoader(my_test_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

my_validation_set = MyDataset(validation_set)
my_validation_set_loader = torch.utils.data.DataLoader(my_validation_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

# DEBUG
if args.debug:
    my_train_set_loader.num_workers = 0
    my_test_set_loader.num_workers = 0
    my_validation_set_loader.num_workers = 0

if __name__ == "__main__":
    plt.ion()
    if args.load_file != False:
        model = load_model_state_dict(args.load_dir+args.load_file)
        my_test("loaded_model", None)
    else:
        loss_plots = {
            "loss": {
                "fig": None,
                "ax": None,
                "line1": None
            },
            "recon_loss": {
                "fig": None,
                "ax": None,
                "line1": None
            },
            "kld_loss": {
                "fig": None,
                "ax": None,
                "line1": None
            },
            "test_loss": {
                "fig": None,
                "ax": None,
                "line1": None
            }
        }
        
        loss_plots["loss"]["fig"] = plt.figure("Loss")
        loss_plots["loss"]["fig"].suptitle('Loss', fontsize=10)
        loss_plots["loss"]["ax"] = loss_plots["loss"]["fig"].add_subplot(1, 1, 1)
        loss_plots["loss"]["line1"], = loss_plots["loss"]["ax"].plot([], [], 'o-r', label="TRAIN Loss")

        # loss_plots["recon_loss"]["fig"] = loss_plots["loss"]["fig"]
        # loss_plots["recon_loss"]["ax"] = loss_plots["loss"]["ax"]
        # loss_plots["recon_loss"]["line1"], = loss_plots["loss"]["ax"].plot([], [], '-g', label="Reconstruction Loss")

        # loss_plots["kld_loss"]["fig"] = loss_plots["loss"]["fig"]
        # loss_plots["kld_loss"]["ax"] = loss_plots["loss"]["ax"]
        # loss_plots["kld_loss"]["line1"], = loss_plots["loss"]["ax"].plot([], [], '-k', label="KLD Loss")

        loss_plots["test_loss"]["fig"] = loss_plots["loss"]["fig"]
        loss_plots["test_loss"]["ax"] = loss_plots["loss"]["ax"]
        loss_plots["test_loss"]["line1"], = loss_plots["loss"]["ax"].plot([], [], 'x-b', label="TEST Loss")

        loss_plots["loss"]["ax"].legend()

        for epoch in range(1, args.epochs + 1):
            my_train(epoch, loss_plots)
            my_test(epoch, loss_plots)

            # random generation of samples to decode
            # with torch.no_grad():
            #     sample = torch.randn(1, 20).to(device)
            #     sample = model.decode(sample).cpu()
                # for i in range(len(sample)):
                #     print (sample[i])
          

        if args.save_file != False:
            save_model_state_dict(args.save_dir+args.save_file)
    
    if args.transformation_plot != False:
        test_data_1 = torch.tensor([])
        test_data_2 = torch.tensor([])
        for batch_idx, (data, label, m, c, index) in enumerate(my_test_set_loader):
            if len(test_data_1) == 0:
                test_data_1 = data[0]
            elif len(test_data_2) == 0:
                test_data_2 = data[0]
            else:
                break
        mu, logvar = model.encode(test_data_1.unsqueeze(0).view(-1, 700))
        test_encoded_1 = model.reparameterize(mu, logvar, False)[0]
        mu, logvar = model.encode(test_data_2.unsqueeze(0).view(-1, 700))
        test_encoded_2 = model.reparameterize(mu, logvar, False)[0]
        steps = int(args.transformation_plot)
        ds = torch.tensor(list(map(lambda e1, e2: (e1 - e2) / steps , test_encoded_1, test_encoded_2)))

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!REMOVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        new_fig = plt.figure()
        # Make a user-defined colormap.
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["b","r"])

        # Make a normalizer that will map the time values from
        # [start_time,end_time+1] -> [0,1].
        cnorm = mcol.Normalize(vmin=0,vmax=steps)

        # Turn these into an object that can be used to map time values to colors and
        # can be passed to plt.colorbar().
        cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
        # cpick = cm.ScalarMappable(norm=cnorm,cmap=plt.get_cmap("autumn"))
        cpick.set_array([])
        
        # F = plt.figure()
        # A = F.add_subplot(111)
        # for y, t in zip(ydat,tim):
        #     A.plot(xdat,y,color=cpick.to_rgba(t))
        initial_color = (0, 0, 1)
        final_color = (1, 0, 0)
        new_color_delta = tuple((item_final - item_initial) / steps for item_initial, item_final in zip(initial_color, final_color))
        new_color = initial_color

        


        for i in range(steps+1):
            decoded = model.decode(test_encoded_2 + i*ds).view(len(test_data_1), -1)
            title = "transformation_plot"
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!REVERT CHANGES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # joints_plot(decoded, i, title)
            # new_color = cpick.to_rgba(i)
            new_joints_plot(decoded, i, title, new_fig, new_color)
            new_color = tuple(current_item + delta_item for current_item, delta_item in zip(new_color, new_color_delta))
        

        # !!!!!!!!!!!!!!!!!!!!!!!!!!REMOVE!!!!!!!!!!!!!!!!!!!!!!!!!!
        # plt.colorbar(cpick,label="Iterations")
        # new_fig.colorbar(cpick,label="Iterations")
        new_fig.subplots_adjust(right=0.8)
        cbar_ax = new_fig.add_axes([0.85, 0.11, 0.02, 0.775])
        new_fig.colorbar(cpick, cax=cbar_ax, label="initial trajectory                                                                                                                            final trajectory", ticks=[])



    if get_encoded_data:
        # global latent_space_train
        data_to_plot = pd.DataFrame()
        test_data_to_plot = pd.DataFrame()
        original_data = pd.Series()
        labels = pd.Series()
        test_labels = pd.Series()
        if latent_space_train == None:
            latent_space_train = torch.tensor([])
            original_data = torch.tensor([])
            # for batch_idx, (data, label) in enumerate(train_loader):
            with torch.no_grad():
                for batch_idx, (data, label, m, c, index) in enumerate(my_train_set_loader):
                    mu, logvar = model.encode(data.view(-1, 700))
                    latent_space_train = torch.cat((latent_space_train, model.reparameterize(mu, logvar, False)), 0)
                    original_data = torch.cat((original_data, data.view(-1, 700)), 0)
                    # labels = labels.append(pd.Series(m), ignore_index=True)
                    labels = labels.append(pd.Series(label), ignore_index=True)
                
                latent_space_test = torch.tensor([])
                for batch_idx, (data, label, m, c, index) in enumerate(my_test_set_loader):
                    mu, logvar = model.encode(data.view(-1, 700))
                    latent_space_test = torch.cat((latent_space_test, model.reparameterize(mu, logvar, False)), 0)
                    # test_labels = test_labels.append(pd.Series(m), ignore_index=True)
                    test_labels = test_labels.append(pd.Series(label), ignore_index=True)
            
            data_to_plot['vel'] = labels
            test_data_to_plot['test_vel'] = test_labels

        if args.tsne != False:
            print ("\nApplying the t-sne algorithm to the latent space train subset...")
            time_start = time.time()
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            tsne_results = tsne.fit_transform(latent_space_train.detach().numpy())
            print('t-SNE over the latent space done! Time elapsed: {} seconds'.format(time.time()-time_start))
            data_to_plot['tsne-train-encoded-' + str(model.fc22.out_features) + 'd-one'] = tsne_results[:,0]
            data_to_plot['tsne-train-encoded-' + str(model.fc22.out_features) + 'd-two'] = tsne_results[:,1]

            # print ("\nApplying the t-sne algorithm to the latent space test subset...")
            # time_start = time.time()
            # tsne_results = tsne.fit_transform(latent_space_test.detach().numpy())
            # print('t-SNE over the latent space done! Time elapsed: {} seconds'.format(time.time()-time_start))
            # test_data_to_plot['tsne-test-encoded-' + str(model.fc22.out_features) + 'd-one'] = tsne_results[:,0]
            # test_data_to_plot['tsne-test-encoded-' + str(model.fc22.out_features) + 'd-two'] = tsne_results[:,1]

            fig = plt.figure("tsne_plot", figsize=(16,10))
            if "BOTH" == str(args.tsne).upper():
                fig.suptitle('t-sne algorithm over ' + str(len(my_train_set_loader.dataset)) + ' trajectories:\nEncoded ' + str(model.fc22.out_features) + ' dimensional data using ' + args.loss_type.upper() + ' loss (left) vs Original ' + str(model.fc1.in_features) + ' dimensional data (right)', fontsize=14)
                print ("\nApplying the t-sne algorithm to the original data subset...")
                time_start = time.time()
                tsne_results = tsne.fit_transform(original_data.detach().numpy())
                print('t-SNE over the original data done! Time elapsed: {} seconds'.format(time.time()-time_start))
                data_to_plot['tsne-original-one'] = tsne_results[:,0]
                data_to_plot['tsne-original-two'] = tsne_results[:,1]
                
                ax0 = plt.subplot(1, 2, 1)
                ax1 = plt.subplot(1, 2, 2)
                g = sns.scatterplot(
                    x="tsne-original-one", y="tsne-original-two",
                    hue="vel",
                    palette=sns.color_palette("hls", data_to_plot['vel'].nunique()),
                    data=data_to_plot,
                    legend="full",
                    alpha=0.3,
                    ax=ax1
                )
                legend = g.legend_
                for i, label_text in enumerate(legend.texts):
                    if i != 0:
                        label_text.set_text(round(float(label_text.get_text()), 1))
            else:
                fig.suptitle('t-sne algorithm over ' + str(len(my_train_set_loader.dataset)) + ' trajectories:\nEncoded ' + str(model.fc22.out_features) + ' dimensional data using ' + args.loss_type.upper() + ' loss', fontsize=14)
                ax0 = plt.subplot(1, 1, 1)
                # ax0 = plt.subplot(1, 2, 1)
                # ax2 = plt.subplot(1, 2, 2)

            ax0.set_title("ALPHA = " + str(args.alpha) + "  BETA = " + str(args.beta))
            g = sns.scatterplot(
                x="tsne-train-encoded-" + str(model.fc22.out_features) + "d-one", y="tsne-train-encoded-" + str(model.fc22.out_features) + "d-two",
                hue="vel",
                palette=sns.color_palette("hls", data_to_plot['vel'].nunique()),
                data=data_to_plot,
                legend="full",
                alpha=0.3,
                ax=ax0
            )
            legend = g.legend_
            for i, label_text in enumerate(legend.texts):
                if i != 0:
                    label_text.set_text(round(float(label_text.get_text()), 1))
            
            # ax2.set_title("ALPHA = " + str(args.alpha) + "  BETA = " + str(args.beta))
            # g = sns.scatterplot(
            #     x="tsne-test-encoded-" + str(model.fc22.out_features) + "d-one", y="tsne-test-encoded-" + str(model.fc22.out_features) + "d-two",
            #     hue="test_vel",
            #     palette=sns.color_palette("hls", test_data_to_plot['test_vel'].nunique()),
            #     data=test_data_to_plot,
            #     legend="full",
            #     alpha=0.3,
            #     ax=ax2
            # )
            # legend = g.legend_
            # for i, label_text in enumerate(legend.texts):
            #     if i != 0:
            #         label_text.set_text(round(float(label_text.get_text()), 1))

        if args.matrix_plot != False:
            dataset_data_to_plot = pd.DataFrame()
            dataset_data_to_plot['vel'] = pd.concat([data_to_plot['vel'], test_data_to_plot['test_vel']], ignore_index=True)
            latent_space_dataset = torch.cat((latent_space_train, latent_space_test), 0)
            latent_space_dimension = len(latent_space_dataset[0])
            for i in range(latent_space_dimension):
                dataset_data_to_plot['latent-space-'+str(i+1)] = [item[i].item() for item in latent_space_dataset]
            fig = plt.figure("matrix_plot", figsize=(16,10))
            fig.suptitle("ALPHA = " + str(args.alpha) + "  BETA = " + str(args.beta))
            for i in range(latent_space_dimension):
                for ii in range(latent_space_dimension):
                    ax0 = plt.subplot(latent_space_dimension, latent_space_dimension, i*latent_space_dimension + ii + 1)
                    a = sns.scatterplot(
                        x="latent-space-"+str(ii+1), y="latent-space-"+str(i+1),
                        hue="vel",
                        palette=sns.color_palette("hls", dataset_data_to_plot['vel'].nunique()),
                        data=dataset_data_to_plot,
                        legend=False,
                        alpha=0.3,
                        ax=ax0
                    )
                    if ii != 0:
                        a.set_ylabel(None)
                    if i != latent_space_dimension-1:
                        a.set_xlabel(None)

    if args.wmb != False:

        def sort_third(vals):
            return vals[2]

        with torch.no_grad():
            train_traj_loss = []
            test_traj_loss = []
            if args.debug:
                temp_train_set_loader = torch.utils.data.DataLoader(my_train_set, batch_size=1, shuffle=True, num_workers=0)
                temp_test_set_loader = torch.utils.data.DataLoader(my_test_set, batch_size=1, shuffle=True, num_workers=0)
            else:
                temp_train_set_loader = torch.utils.data.DataLoader(my_train_set, batch_size=1, shuffle=True, num_workers=4)
                temp_test_set_loader = torch.utils.data.DataLoader(my_test_set, batch_size=1, shuffle=True, num_workers=4)
            
            print ("Scanning the dataset...")
            for batch_idx, (data, label, m, c, index) in enumerate(temp_train_set_loader):
                recon_batch, mu, logvar = model(data, False)
                loss, recon_loss, kld_loss = loss_function(recon_batch, data, mu, logvar, label)
                train_traj_loss.append((data[0], recon_batch.view(len(data[0]), -1), loss))

            for batch_idx, (data, label, m, c, index) in enumerate(temp_test_set_loader):
                recon_batch, mu, logvar = model(data, False)
                loss, recon_loss, kld_loss = loss_function(recon_batch, data, mu, logvar, label)
                test_traj_loss.append((data[0], recon_batch.view(len(data[0]), -1), loss))
            
            train_traj_loss.sort(key=sort_third)
            test_traj_loss.sort(key=sort_third)

            title = "Train sample with the lowest erorr\nError = "+str(train_traj_loss[0][2].item())
            joints_plot(train_traj_loss[0][0], train_traj_loss[0][1], title)
            title = "Train sample with the median erorr\nError = "+str(train_traj_loss[int(len(train_traj_loss)/2)][2].item())
            joints_plot(train_traj_loss[int(len(train_traj_loss)/2)][0], train_traj_loss[int(len(train_traj_loss)/2)][1], title)
            title = "Train sample with the highest erorr\nError = "+str(train_traj_loss[-1][2].item())
            joints_plot(train_traj_loss[-1][0], train_traj_loss[-1][1], title)
            
            title = "Test sample with the lowest erorr\nError = "+str(test_traj_loss[0][2].item())
            joints_plot(test_traj_loss[0][0], test_traj_loss[0][1], title)
            title = "Test sample with the median erorr\nError = "+str(test_traj_loss[int(len(test_traj_loss)/2)][2].item())
            joints_plot(test_traj_loss[int(len(test_traj_loss)/2)][0], test_traj_loss[int(len(test_traj_loss)/2)][1], title)
            title = "Test sample with the highest erorr\nError = "+str(test_traj_loss[-1][2].item())
            joints_plot(test_traj_loss[-1][0], test_traj_loss[-1][1], title)

    if args.write != False:
        True

    plt.show()

    input("Press Enter to close...")