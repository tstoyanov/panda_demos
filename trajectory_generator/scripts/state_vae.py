import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset

import random, json, ast, os, copy, time
from os.path import isfile

import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import pandas as pd

from collections import Counter


import importlib
from nn_models.model_state_vae import STATE_VAE as STATE_VAE

import rospkg
rospack = rospkg.RosPack()
package_path = rospack.get_path("trajectory_generator")

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='state_vae')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train for (default: 10)')
parser.add_argument('--learning-rate', type=float, default=0.001,
                    help='value of the learning rate of the optimizer (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--dataset-dir', default="latest",
                    help='path of the directory containing the input dataset')
parser.add_argument('--debug', nargs='?', const=True, default=False,
                    help='debug flag')
parser.add_argument('--save-dir', default=package_path + "/saved_models/state_vae/",
                    help='directory where to save the model once trained')
parser.add_argument('--save-file', default=True,
                    help='name of the file to save the model once trained')
parser.add_argument('--load-dir', default=package_path + "/saved_models/state_vae/",
                    help='directory from where to load the trained model')
parser.add_argument('--load-file', default=False,
                    help='name of the file to load the trained model from')
parser.add_argument('--loss-type', default="MSE_SUM",
                    help='type of loss metric to combine with KLD while calculating the loss')
parser.add_argument('--alpha', default=1,
                    help='value of the recostruction loss weight in loss calculation')
parser.add_argument('--beta', default=1,
                    help='value of the KLD weight in loss calculation')
parser.add_argument('--test-percentage', default=10,
                    help='percentage of the dataset to use as test data (0, 100)')
parser.add_argument('--validation-percentage', default=10,
                    help='percentage of the dataset to use as validation data (0, 100)')
parser.add_argument('--random', nargs='?', const=True, default=False,
                    help='whether to randomize the test set initialization or not')
parser.add_argument('--vae-dim', default=4,
                    help='set the dimension of the latent space of the VAE used to encode the state')
parser.add_argument('--ioff', nargs='?', const=True, default=False,
                    help='disables interactive plotting of the train and test error')
parser.add_argument('--parameters-search', nargs='?', const=True, default=False,
                    help='wether to perform parameter search or not')

args, unknown = parser.parse_known_args()

if args.dataset_dir[0] != "/":
    args.dataset_dir = "/" + args.dataset_dir
if args.dataset_dir[-1] != "/":
    args.dataset_dir = args.dataset_dir + "/"
args.dataset_dir = package_path + "/sensing_data" + args.dataset_dir
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

class stones_dataset(Dataset):
    """
    This dataset contains the force readings stored in the file passed as argument
    The __getitem__ method returns forces and torques in the three axes
    """
    def __init__(self, input_dataset):
        super(stones_dataset, self).__init__()
        self.dataset = input_dataset
        
    def __len__(self):
        return len(self.dataset["force"]["x"])
        
    def __getitem__(self, index):
        assert 0 <= index < len(self.dataset["force"]["x"])
        ret = []
        ret.append(self.dataset["force"]["x"][index])
        ret.append(self.dataset["force"]["y"][index])
        ret.append(self.dataset["force"]["z"][index])
        ret.append(self.dataset["torque"]["x"][index])
        ret.append(self.dataset["torque"]["y"][index])
        ret.append(self.dataset["torque"]["z"][index])
        return torch.tensor(ret), torch.tensor(float(self.dataset["label"][index])), index

def update_graph(fig, ax, line1, x_value, y_value):
    line1.set_xdata(np.append(line1.get_xdata(), x_value))
    line1.set_ydata(np.append(line1.get_ydata(), y_value))
    # plt.draw()

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

def shuffle_lists(*args):
    for i in reversed(range(1, len(args[0]))):
        j = int(random.random() * (i+1))
        for x in args: 
            x[i], x[j] = x[j], x[i]

def data_split(input_dataset, test_percentage, validation_percentage):
    input_train_set = input_dataset + "train_set.txt"
    input_test_set = input_dataset + "test_set.txt"
    input_validation_set = input_dataset + "validation_set.txt"
    input_dataset += "stone_dataset.txt"
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
        dataset_length = len(train_set["force"]["x"])
        test_set = {
            "force": {
                "x": [],
                "y": [],
                "z": []
            },
            "torque": {
                "x": [],
                "y": [],
                "z": []
            },
            "label": []
        }
        validation_set = {
            "force": {
                "x": [],
                "y": [],
                "z": []
            },
            "torque": {
                "x": [],
                "y": [],
                "z": []
            },
            "label": []
        }

        # random test set initialization
        if args.random == True:
            for i in range(int(dataset_length * float(test_percentage) / 100)):
                sample_index = random.randint(0,len(train_set["force"]["x"])-1)
                for key in train_set["force"]:
                    test_set["force"][key].append(train_set["force"][key].pop(sample_index))
                    test_set["torque"][key].append(train_set["force"][key].pop(sample_index))
                test_set["label"].append(train_set["label"].pop(sample_index))
            for i in range(int(dataset_length * float(validation_percentage) / 100)):
                sample_index = random.randint(0,len(train_set["force"][key])-1)
                for key in train_set["force"]:
                    validation_set["force"][key].append(train_set["force"][key].pop(sample_index))
                    validation_set["torque"][key].append(train_set["force"][key].pop(sample_index))
                validation_set["label"].append(train_set["label"].pop(sample_index))

        # non-random test set initialization
        else:
            shuffle_lists(test_set["force"]["x"], test_set["force"]["y"], test_set["force"]["z"], test_set["torque"]["x"], test_set["torque"]["y"], test_set["torque"]["z"], test_set["label"])

            subsets_composition = list(map(lambda item: ((item[0], int(item[1]*test_percentage/100)), (item[0], int(item[1]*validation_percentage/100))), list(Counter(train_set["label"]).items())))
            test_composition = [item[0] for item in subsets_composition]
            validation_composition = [item[1] for item in subsets_composition]
            
            for (label, amount) in test_composition:
                for i in range(amount):
                    try:
                        sample_index = train_set["label"].index(label)
                        for key in train_set["force"]:
                            test_set["force"][key].append(train_set["force"][key].pop(sample_index))
                            test_set["torque"][key].append(train_set["force"][key].pop(sample_index))
                        test_set["label"].append(train_set["label"].pop(sample_index))
                    except:
                        break
            while len(test_set["force"]["x"]) < dataset_length * test_percentage / 100:
                sample_index = random.randint(0,len(train_set["force"]["x"])-1)
                for key in train_set["force"]:
                    test_set["force"][key].append(train_set["force"][key].pop(sample_index))
                    test_set["torque"][key].append(train_set["force"][key].pop(sample_index))
                test_set["label"].append(train_set["label"].pop(sample_index))
            
            for (label, amount) in validation_composition:
                for i in range(amount):
                    try:
                        sample_index = train_set["label"].index(label)
                        for key in train_set["force"]:
                            validation_set["force"][key].append(train_set["force"][key].pop(sample_index))
                            validation_set["torque"][key].append(train_set["force"][key].pop(sample_index))
                        validation_set["label"].append(train_set["label"].pop(sample_index))
                    except:
                        break
            while len(validation_set["force"]["x"]) < dataset_length * validation_percentage / 100:
                sample_index = random.randint(0,len(train_set["force"]["x"])-1)
                for key in train_set["force"]:
                    validation_set["force"][key].append(train_set["force"][key].pop(sample_index))
                    validation_set["torque"][key].append(train_set["force"][key].pop(sample_index))
                validation_set["label"].append(train_set["label"].pop(sample_index))
        
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

def loss_function(recon_x, x, mu, logvar, label):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    if "MSE_MEAN" == args.loss_type.upper():
        criterion = nn.MSELoss(reduction='mean')
        reconstruction_loss = criterion(recon_x, x)
        return float(args.alpha)*reconstruction_loss + float(args.beta)*KLD, float(args.alpha)*reconstruction_loss, float(args.beta)*KLD
    if "MSE_SUM" == args.loss_type.upper():
        criterion = nn.MSELoss(reduction='sum')
        reconstruction_loss = criterion(recon_x, x)
        return float(args.alpha)*reconstruction_loss + float(args.beta)*KLD, float(args.alpha)*reconstruction_loss, float(args.beta)*KLD
    if "MSE_NONE" == args.loss_type.upper():
        criterion = nn.MSELoss(reduction='none')
        reconstruction_loss = criterion(recon_x, x)
        return float(args.alpha)*reconstruction_loss + float(args.beta)*KLD, float(args.alpha)*reconstruction_loss, float(args.beta)*KLD

    elif "BCE_SUM" == args.loss_type.upper():
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        return float(args.alpha)*BCE + float(args.beta)*KLD
    elif "BCE_MEAN" == args.loss_type.upper():
        BCE = F.binary_cross_entropy(recon_x, x, reduction='mean')
        return float(args.alpha)*BCE + float(args.beta)*KLD
    elif "BCE_NONE" == args.loss_type.upper():
        BCE = F.binary_cross_entropy(recon_x, x, reduction='none')
        return float(args.alpha)*BCE + float(args.beta)*KLD

    elif "L1_NORM_MEAN" == args.loss_type.upper() or "MAE_MEAN" == args.loss_type.upper() or "L1_MEAN" == args.loss_type.upper():
        criterion = nn.L1Loss(reduction="mean")
        reconstruction_loss = criterion(recon_x, x)
        return float(args.alpha)*reconstruction_loss + float(args.beta)*KLD, float(args.alpha)*reconstruction_loss, float(args.beta)*KLD
    elif "L1_NORM_SUM" == args.loss_type.upper() or "MAE_SUM" == args.loss_type.upper() or "L1_SUM" == args.loss_type.upper():
        criterion = nn.L1Loss(reduction="sum")
        reconstruction_loss = criterion(recon_x, x)
        return float(args.alpha)*reconstruction_loss + float(args.beta)*KLD, float(args.alpha)*reconstruction_loss, float(args.beta)*KLD
    elif "L1_NORM_NONE" == args.loss_type.upper() or "MAE_NONE" == args.loss_type.upper() or "L1_NONE" == args.loss_type.upper():
        criterion = nn.L1Loss(reduction="none")
        reconstruction_loss = criterion(recon_x, x)
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

def save_model_state_dict(save_path):
    torch.save(vae_model.state_dict(), save_path)

def load_model_state_dict(load_path):
    model_sd = torch.load(load_path, map_location=torch.device('cpu'))
    loaded_model = STATE_VAE(len(model_sd["fc21.bias"])).to(device)
    loaded_model.load_state_dict(model_sd)
    loaded_model.eval()
    return loaded_model

def my_train(epoch, loss_plots):

    vae_model.train()
    # print ("model: ", model)
    train_loss = 0
    for batch_idx, (data, label, index) in enumerate(my_train_set_loader):
        data = data.to(device)
        vae_optimizer.zero_grad()
        recon_batch, mu, logvar = vae_model(data.view(-1, 600), False)
        # print ("mu.grad_fn: ", mu.grad_fn)
        # print ("mu.grad_fn.next_functions[0][0]: ", mu.grad_fn.next_functions[0][0])
        # print ("logvar.grad_fn: ", logvar.grad_fn)
        loss, recon_loss, kld_loss = loss_function(recon_batch, data.view(-1, 600), mu, logvar, label)
        # print ("value of model.fc1.bias.grad before backward: ")
        # print (model.fc1.bias.grad)
        loss.backward()
        train_loss += loss.item()
        vae_optimizer.step()
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

        if args.batch_size == len(my_train_set) or (batch_idx % args.log_interval == 0 and epoch != 1):
            update_graph(loss_plots["loss"]["fig"], loss_plots["loss"]["ax"], loss_plots["loss"]["line1"], batch_idx + epoch*len(my_train_set_loader), loss.item()/len(data))
        #     update_graph(loss_plots["recon_loss"]["fig"], loss_plots["recon_loss"]["ax"], loss_plots["recon_loss"]["line1"], batch_idx + epoch*len(my_train_set_loader), recon_loss.item()/len(data))
        #     update_graph(loss_plots["kld_loss"]["fig"], loss_plots["kld_loss"]["ax"], loss_plots["kld_loss"]["line1"], batch_idx + epoch*len(my_train_set_loader), kld_loss.item()/len(data))

    # update_graph(loss_plots["loss"]["fig"], loss_plots["loss"]["ax"], loss_plots["loss"]["line1"], len(my_train_set_loader) + epoch*len(my_train_set_loader), loss.item()/len(data))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(my_train_set_loader.dataset)))
    
    latest_train_loss = train_loss / len(my_train_set_loader.dataset)

def my_test(epoch, loss_plots):
    vae_model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, label, index) in enumerate(my_test_set_loader):
            data = data.to(device)
            recon_batch, mu, logvar = vae_model(data.view(-1, 600), False)
            test_loss += loss_function(recon_batch, data.view(-1, 600), mu, logvar, label)[0].item()

            # if batch_idx == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            #     save_image(comparison.cpu(),
            #              'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(my_test_set_loader.dataset)
    args.latest_test_loss = copy.deepcopy(test_loss)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    if loss_plots != None:
        update_graph(loss_plots["test_loss"]["fig"], loss_plots["test_loss"]["ax"], loss_plots["test_loss"]["line1"], len(my_train_set_loader) + epoch*len(my_train_set_loader), test_loss)
    # update_graph(loss_plots["test_loss"]["fig"], loss_plots["test_loss"]["ax"], loss_plots["test_loss"]["line1"], len(my_train_set_loader) + epoch*len(my_train_set_loader), test_loss.item()/len(data))

train_set, test_set, validation_set = data_split(args.dataset_dir, args.test_percentage, args.validation_percentage)

my_train_set = stones_dataset(train_set)
my_train_set_loader = torch.utils.data.DataLoader(my_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

my_test_set = stones_dataset(test_set)
my_test_set_loader = torch.utils.data.DataLoader(my_test_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

my_validation_set = stones_dataset(validation_set)
my_validation_set_loader = torch.utils.data.DataLoader(my_validation_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

# DEBUG
if args.debug:
    my_train_set_loader.num_workers = 0
    my_test_set_loader.num_workers = 0
    my_validation_set_loader.num_workers = 0


if __name__ == "__main__":
    if args.ioff != True:
        plt.ion()
    if args.load_file != False:
        vae_model = load_model_state_dict(args.load_dir+args.load_file)
        my_test("loaded_model", None)
    else:
        if args.parameters_search == True:
            alpha_list = [1, 2, 5, 10, 25, 50, 100, 250, 500, 100, 200, 50, 100, 1, 1, 1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.01, 0.01]
            beta_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.01, 0.01, 2, 5, 10, 25, 50, 100, 250, 500, 100, 200, 50, 100]
            # alpha_list = [200, 100]
            # beta_list = [0.1, 0.1]
        else:
            alpha_list = [args.alpha]
            beta_list = [args.beta]

        for i, _ in enumerate(alpha_list):
            vae_model = STATE_VAE(int(args.vae_dim)).to(device)
            vae_optimizer = optim.Adam(vae_model.parameters(), lr=args.learning_rate)

            args.alpha = alpha_list[i]
            args.beta = beta_list[i]

            args.latest_train_loss = None
            args.latest_test_loss = None
            best_train_loss = None
            best_test_loss = None

            head, tail = os.path.split(args.dataset_dir)
            head, tail = os.path.split(head)
            dataset_str = tail

            alpha = float(args.alpha)
            beta = float(args.beta)
            e_alpha = 0
            e_beta = 0
            while alpha < 1:
                alpha *= 10
                e_alpha += 1
            alpha = int(alpha)
            if e_alpha > 0:
                alpha_str = "a"+str(alpha)+"e-"+str(e_alpha)
            else:
                alpha_str = "a"+str(alpha)
            while beta < 1:
                beta *= 10
                e_beta += 1
            beta = int(beta)
            if e_beta > 0:
                beta_str = "b"+str(beta)+"e-"+str(e_beta)
            else:
                beta_str = "b"+str(beta)
            save_path = args.save_dir + str(args.vae_dim) + "_dim/" + alpha_str + "_" + beta_str + "_" + str(args.epochs) + "e_" + dataset_str + "/model.pt"
            # if int(args.vae_dim) != 5:
            #     save_path = args.save_dir + str(args.vae_dim) + "_dim/" + alpha_str + "_" + beta_str + "_" + str(args.epochs) + "e_" + dataset_str + "/model.pt"
            # else:
            #     save_path = args.save_dir + alpha_str + "_" + beta_str + "_" + str(args.epochs) + "e_" + dataset_str + "/model.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

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
                if epoch == 1:
                    best_test_loss = args.latest_test_loss
                elif args.latest_test_loss < best_test_loss:
                    save_model_state_dict(os.path.dirname(save_path) + "/best_model.pt")
                    best_test_loss = args.latest_test_loss
            

            if args.save_file != False:
                save_model_state_dict(save_path)
                # save_model_state_dict(args.save_dir+args.save_file)
            
            if args.parameters_search != False:
                save_model_state_dict(save_path)
                loss_plots["loss"]["ax"].set_title("ALPHA = " + str(args.alpha) + "   BETA = " + str(args.beta))
                loss_plots["loss"]["fig"].savefig(os.path.dirname(save_path) + "/loss.svg", format="svg", bbox_inches='tight')
                loss_plots["loss"]["fig"].savefig(os.path.dirname(save_path) + "/loss.png", format="png", bbox_inches='tight')
                plt.close(loss_plots["loss"]["fig"])