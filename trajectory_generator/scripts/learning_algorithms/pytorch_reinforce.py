import argparse
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=12, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--epochs', type=int, default=20000,
                    help='number of epochs for training (default: 20)')
parser.add_argument('--batch-size', type=int, default=12, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--state-dim', type=int, default=4,
                    help='policy input dimension (default: 4)')
parser.add_argument('--action-dim', type=int, default=5,
                    help='policy output dimension (default: 5)')
parser.add_argument('--learning-rate', type=float, default=0.001,
                    help='learning rate of the optimizer')
parser.add_argument('--no-plot', nargs='?', const=True, default=False,
                    help='whether to plot data or not')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
args.plot = not args.no_plot
# torch.manual_seed(args.seed)

class Policy(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Policy, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(self.in_dim, 24)
        
        self.fc21 = nn.Linear(24, 24)  # mean layer
        self.fc31 = nn.Linear(24, self.out_dim)

        self.fc22 = nn.Linear(24, 24)  # log_var layer
        self.fc32 = nn.Linear(24, self.out_dim)

        # episode data
        self.saved_log_probs = []
        self.rewards = []
        
        # history data
        self.actions_history = []
        self.log_probs_history = []
        self.rewards_history = []
        self.deterministic_policy_rewards_history = []
        self.deterministic_policy_means_history = []
        self.episode_rewards_history = []
        self.means_history = []
        self.losses_history = []
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        
        h21 = self.fc21(h)
        mean = self.fc31(h21)

        h22 = self.fc22(h)
        log_var = self.fc32(h22)

        return mean, log_var

    def forward(self, x):
        return self.encode(x)


class ALGORITHM:
    def __init__(self, state_dim, action_dim, lr, plot=True, batch_size=None):
        self.live_plots = {
            "loss": {
                "fig": None,
                "ax": None,
                "line1": None
            },
            "reward": {
                "fig": None,
                "ax": None,
                "line1": None
            },
            "theta": {
                "fig": [],
                "ax": [],
                "lines": []
            }
        }
        self.lr = lr
        self.plot = plot
        self.current_epoch = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy = Policy(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.eps = np.finfo(np.float32).eps.item()
        self.stones_positions = {
            "distance": [],
            "angle": [],
        }
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = args.batch_size

        if self.plot:
            plt.ion()
            self.live_plots["loss"]["fig"] = plt.figure("Loss")
            self.live_plots["loss"]["ax"] = self.live_plots["loss"]["fig"].add_subplot(2, 1, 1)
            self.live_plots["loss"]["line1"], = self.live_plots["loss"]["ax"].plot([], [], 'o-r', label="TRAIN Loss")

            self.live_plots["reward"]["fig"] = self.live_plots["loss"]["fig"]
            self.live_plots["reward"]["ax"] = self.live_plots["loss"]["fig"].add_subplot(2, 1, 2)
            self.live_plots["reward"]["line1"], = self.live_plots["reward"]["ax"].plot([], [], 'o-b', label="Reward")

            for i in range(self.action_dim):
                self.live_plots["theta"]["fig"].append(plt.figure("theta-"+str(i), figsize=(6, 2)))
                self.live_plots["theta"]["ax"].append(self.live_plots["theta"]["fig"][-1].add_subplot(1, 1, 1))
                self.live_plots["theta"]["lines"].append(self.live_plots["theta"]["ax"][-1].plot([], [], label="["+str(i)+"]", marker="o")[0])
            # self.live_plots["theta"]["ax"].axhline(y=0, color="k")


            self.live_plots["loss"]["ax"].legend()
            self.live_plots["reward"]["ax"].legend()
            # self.live_plots["theta"]["ax"].legend()
    
    def save_model_state_dict(self, save_path):
        torch.save(self.policy.state_dict(), save_path)

    def load_model_state_dict(self, load_path):
        model_sd = torch.load(load_path)
        loaded_model = Policy(self.state_dim, self.action_dim)
        loaded_model.load_state_dict(model_sd)
        loaded_model.eval()
        self.policy = loaded_model
        return loaded_model
        # return VAE().to(device).load_state_dict(torch.load(load_path)).eval()
    
    def save_checkpoint(self, save_path):
        torch.save({
            "algorithm": {
                'model_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                "lr": self.lr,
                "current_epoch": self.current_epoch,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "eps": self.eps,
                "batch_size": self.batch_size,
                "stones_positions": self.stones_positions,
            },
            "policy": {
                "in_dim": self.policy.in_dim,
                "out_dim": self.policy.out_dim,

                "saved_log_probs": self.policy.saved_log_probs,
                "rewards": self.policy.rewards,
                
                "actions_history": self.policy.actions_history,
                "log_probs_history": self.policy.log_probs_history,
                "rewards_history": self.policy.rewards_history,
                "deterministic_policy_rewards_history": self.policy.deterministic_policy_rewards_history,
                "deterministic_policy_means_history": self.policy.deterministic_policy_means_history,
                "episode_rewards_history": self.policy.episode_rewards_history,
                "means_history": self.policy.means_history,
                "losses_history": self.policy.losses_history,
            }
        }, save_path)

    def load_checkpoint(self, load_path):
        checkpoint = torch.load(load_path)
        if checkpoint.has_key("algorithm"):
            if checkpoint["policy"].has_key("lr"):
                self.lr = checkpoint["algorithm"]["lr"]
            if checkpoint["policy"].has_key("current_epoch"):
                self.current_epoch = checkpoint["algorithm"]["current_epoch"]
            if checkpoint["policy"].has_key("state_dim"):
                self.state_dim = checkpoint["algorithm"]["state_dim"]
            if checkpoint["policy"].has_key("action_dim"):
                self.action_dim = checkpoint["algorithm"]["action_dim"]
            if checkpoint["policy"].has_key("eps"):
                self.eps = checkpoint["algorithm"]["eps"]
            if checkpoint["policy"].has_key("batch_size"):
                self.batch_size = checkpoint["algorithm"]["batch_size"]
            if checkpoint["policy"].has_key("stones_positions"):
                self.stones_positions = checkpoint["algorithm"]["stones_positions"]
            
            if checkpoint["policy"].has_key("model_state_dict"):
                self.policy.load_state_dict(checkpoint["algorithm"]['model_state_dict'])
            if checkpoint["policy"].has_key("optimizer_state_dict"):
                self.optimizer.load_state_dict(checkpoint["algorithm"]['optimizer_state_dict'])

        if checkpoint.has_key("policy"):
            if checkpoint["policy"].has_key("in_dim"):
                self.policy.in_dim = checkpoint["policy"]["in_dim"]
            if checkpoint["policy"].has_key("out_dim"):
                self.policy.out_dim = checkpoint["policy"]["out_dim"]
            if checkpoint["policy"].has_key("saved_log_probs"):
                self.policy.saved_log_probs = checkpoint["policy"]["saved_log_probs"]
            if checkpoint["policy"].has_key("rewards"):
                self.policy.rewards = checkpoint["policy"]["rewards"]      
            if checkpoint["policy"].has_key("actions_history"):
                self.policy.actions_history = checkpoint["policy"]["actions_history"]
            if checkpoint["policy"].has_key("log_probs_history"):
                self.policy.log_probs_history = checkpoint["policy"]["log_probs_history"]
            if checkpoint["policy"].has_key("rewards_history"):
                self.policy.rewards_history = checkpoint["policy"]["rewards_history"]
            if checkpoint["policy"].has_key("deterministic_policy_rewards_history"):
                self.policy.deterministic_policy_rewards_history = checkpoint["policy"]["deterministic_policy_rewards_history"]
            if checkpoint["policy"].has_key("deterministic_policy_means_history"):
                self.policy.deterministic_policy_means_history = checkpoint["policy"]["deterministic_policy_means_history"]
            if checkpoint["policy"].has_key("episode_rewards_history"):
                self.policy.episode_rewards_history = checkpoint["policy"]["episode_rewards_history"]
            if checkpoint["policy"].has_key("means_history"):
                self.policy.means_history = checkpoint["policy"]["means_history"]
            if checkpoint["policy"].has_key("losses_history"):
                self.policy.losses_history = checkpoint["policy"]["losses_history"]


        self.policy.train()

    def select_action(self, state, cov_mat=None, target_action=None):
        mean, log_var = self.policy(state)
        std = torch.exp(0.5*log_var)
        # print ("mean = {}".format(mean.data))
        # print ("mean = {} - var = {}".format(mean.data, torch.exp(log_var).data))

        if cov_mat is not None:
            cov_matrix = cov_mat
        else:
            cov_matrix = torch.diag(torch.tensor([0.001]*self.policy.out_dim))

        # cov_matrix = torch.diag(torch.tensor([0]*self.policy.out_dim))
        # cov_matrix = torch.diag(torch.exp(log_var))
        dist = MultivariateNormal(mean, cov_matrix)


        if target_action is not None:
            action_sample = target_action
        else:
            action_sample = dist.sample()

        log_prob = dist.log_prob(action_sample)
        self.policy.saved_log_probs.append(log_prob)

        self.policy.actions_history.append(action_sample)
        self.policy.log_probs_history.append(log_prob)
        self.policy.means_history.append(mean)
        return action_sample, mean
    
    def get_policy_mean(self, state):
        mean, log_var = self.policy(state)
        return mean


    def finish_episode(self, pre_training=False):
        R = 0
        policy_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            # R = r + args.gamma * R
            # returns.insert(0, R)
            returns.insert(0, r)
        returns = torch.FloatTensor(returns)
        # print ("batch mean = {} . batch std = {}".format(returns.mean(), returns.std()))
        # returns = (returns - returns.mean()) # baseline
        returns = (returns - returns.mean()) / (returns.std() + self.eps) # baseline and normalization
        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.policy.losses_history.append(policy_loss)
        for i in range(1):
            self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]
        self.current_epoch += 1
        return policy_loss

    def exp_lr_scheduler(self, optimizer, epoch, lr_decay=0.1, lr_decay_epoch=2):
        """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
        if epoch % lr_decay_epoch:
            return optimizer
        
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        return optimizer

    def set_reward(self, reward):
        self.policy.rewards.append(reward)
        self.policy.rewards_history.append(reward)
    
    def set_deterministic_policy_reward(self, reward):
        self.policy.deterministic_policy_rewards_history.append(reward)
    
    def set_deterministic_policy_mean(self, reward):
        self.policy.deterministic_policy_means_history.append(reward)
    
    def set_episode_reward(self, reward):
        self.policy.episode_rewards_history.append(reward)

    def set_stone_position(self, distance, angle):
        self.stones_positions["distance"].append(distance)
        self.stones_positions["angle"].append(angle)

    def update(self):
        True

    def update_graph(self, fig, ax, line, x_value, y_value):
        line.set_xdata(np.append(line.get_xdata(), x_value))
        line.set_ydata(np.append(line.get_ydata(), y_value))
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    def update_graphs(self):
        if len(self.policy.losses_history) > 0:
            self.update_graph(self.live_plots["loss"]["fig"], self.live_plots["loss"]["ax"], self.live_plots["loss"]["line1"], (self.current_epoch)*self.batch_size, self.policy.losses_history[-1].item())
        if len(self.policy.episode_rewards_history) > 0:
            self.update_graph(self.live_plots["reward"]["fig"], self.live_plots["reward"]["ax"], self.live_plots["reward"]["line1"], (self.current_epoch)*self.batch_size, self.policy.episode_rewards_history[-1])
        if len(self.policy.means_history) > 0:
            for i in range(self.action_dim):
                self.update_graph(self.live_plots["theta"]["fig"][i], self.live_plots["theta"]["ax"][i], self.live_plots["theta"]["lines"][i], (self.current_epoch)*self.batch_size, self.policy.means_history[-1][i].item())
    
    def execute_action(self, action, target=None):
        error = False
        # if all(action < 0.2) and all(action > -0.2):
        #     reward = 1
        # else:
        #     reward = -1
        if target is not None:
            reward = -(abs(action-target)).sum().pow(2).item() * 100
        else:
            reward = -(abs(action)).sum().pow(2).item() * 100

        return error, reward

    def pre_train(self, epochs, batch_size, log_interval, target=None, target_action=None):
        for epoch in range(epochs):
            reward = 0
            for t in range(0, batch_size):  # Don't infinite loop while learning

                starting_state = torch.ones(self.policy.in_dim)
                # starting_state = torch.zeros(self.policy.in_dim)
                # starting_state = torch.randn(self.policy.in_dim)
                if target_action is not None:
                    action, mean = self.select_action(starting_state, target_action=target_action)
                else:
                    action, mean = self.select_action(starting_state)
                error, reward = self.execute_action(action, target=target)
                self.policy.rewards.append(reward)
                self.policy.rewards_history.append(reward)

            loss = self.finish_episode(pre_training=True)

            if epoch % log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}'.format(
                    epoch, reward))
                if self.plot:
                    self.update_graphs()
            
            self.current_epoch += 1

        del self.policy.actions_history[:]
        del self.policy.log_probs_history[:]
        del self.policy.rewards_history[:]
        del self.policy.means_history[:]
        del self.policy.losses_history[:]
        self.current_epoch = 0
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

    def close(self):
        plt.close("all")

def main(args):

    algorithm = ALGORITHM()
    starting_state = torch.ones(algorithm.policy.in_dim)
    mean = None
    for epoch in range(args.epochs):
        reward = 0
        for t in range(0, args.batch_size):  # Don't infinite loop while learning

            starting_state = torch.randn(algorithm.policy.in_dim)
            action, mean = algorithm.select_action(starting_state)
            error, reward = algorithm.execute_action(action)
            algorithm.policy.rewards.append(reward)
            algorithm.policy.rewards_history.append(reward)

        loss = algorithm.finish_episode()
        # exp_lr_scheduler(optimizer, epoch)
        if algorithm.current_epoch % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}'.format(
                  algorithm.current_epoch, reward))
            algorithm.update_graphs()
        algorithm.current_epoch += 1
    
    # plt.show()


if __name__ == '__main__':
    main(args)
