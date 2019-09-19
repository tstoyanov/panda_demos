#!/usr/bin/env python
import sys
import torch
import argparse
import importlib

import rospkg
rospack = rospkg.RosPack()
package_path = rospack.get_path("trajectory_generator")

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs for training (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--state-dim', type=int, default=4,
                    help='policy input dimension (default: 4)')
parser.add_argument('--action-dim', type=int, default=5,
                    help='policy output dimension (default: 5)')
parser.add_argument('--learning-rate', type=int, default=0.002,
                    help='learning rate of the optimizer')
parser.add_argument('--models-dir', default="nn_models",
                    help='directory from where to load the network shape of the action decoder')
parser.add_argument('--decoder-model-file', default="model_trajectory_vae",
                    help='file from where to load the network shape of the action decoder')
parser.add_argument('--decoder-dir', default=package_path + "/saved_models/trajectory_vae/",
                    help='directory from where to load the trained model of the action decoder')
parser.add_argument('--decoder-sd', default=False,
                    help='file from where to load the trained model of the action decoder')
parser.add_argument('--encoder-dir', default=package_path + "/saved_models/state_vae/",
                    help='directory from where to load the trained model of the state encoder')
parser.add_argument('--encoder-file', default="model_state_vae.py",
                    help='file from where to load the network shape of the state encoder')
parser.add_argument('--encoder-sd', default=False,
                    help='file from where to load the trained model of the state encoder')
parser.add_argument('--algorithm-dir', default="learning_algorithms",
                    help='directory from where to load the learning algorithm')
parser.add_argument('--algorithm', default="pytorch_reinforce",
                    help='file from where to load the learning algorithm')
parser.add_argument('--scripts-dir', default=package_path + "/scripts/",
                    help='directory from where to load the scripts')
parser.add_argument('--image-reader', default="imager",
                    help='file from where to load the learning algorithm')
parser.add_argument('--action-script', default="writer_from_generated",
                    help='file from where to load the learning algorithm')
parser.add_argument('--no-plot', nargs='?', const=True, default=False,
                    help='whether to plot data or not')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if not args.decoder_sd:
    print ("No decoder state dictionary specified: provide the file name of the decoder trained model using the '--decoder-sd' argument")
    sys.exit(2)
else:
    decoder_sd = torch.load(args.decoder_dir+args.decoder_sd)
    args.action_dim = len(decoder_sd["fc21.bias"])
    decoder_module = importlib.import_module(args.models_dir + "." + args.decoder_model_file)
    decoder_model = decoder_module.VAE(args.action_dim).to(device)
    decoder_model.load_state_dict(decoder_sd)
    decoder_model.eval()

if not args.algorithm:
    print ("No learning algorithm specified: provide the file name of the learning algorithm using the '--algorithm' argument")
    sys.exit(2)
else:
    algorithm_module = importlib.import_module(args.algorithm_dir + "." + args.algorithm)
    algorithm = algorithm_module.ALGORITHM()

action_script = importlib.import_module(args.scripts_dir + "." + args.action_script)
image_reader_module = importlib.import_module(args.scripts_dir + "." + args.image_reader)

def get_dummy_state():
    return torch.ones(algorithm.policy.in_dim)

def execute_action(action):
    True
    # action_script

def main(args):
    algorithm_module.main(args)
    # image_reader.initialize()
    # state = get_dummy_state()
    # action = algorithm.select_action(state)


if __name__ == '__main__':
    main(args)