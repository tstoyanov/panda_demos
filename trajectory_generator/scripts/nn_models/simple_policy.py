import math
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

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

        self.p1 = torch.nn.Parameter(torch.randn(1))
        self.p2 = torch.nn.Parameter(torch.randn(1))

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
        mean = torch.nn.Parameter(torch.Tensor([self.p1, self.p2]))
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
        print (action_sample)
        log_prob = dist.log_prob(action_sample)
        # =============== NEW CODE ===============
        
        # =============== OLD CODE ===============
        # action_sample = self.sample_action(mean, log_var, no_noise).data
        # action_prob = self.normal_prob(action_sample, mean, sigma_sq)
        # log_prob = action_prob.log()
        # =============== OLD CODE ===============
        entropy = -0.5*((sigma_sq+2*pi.expand_as(sigma_sq)).log()+1)

        return action_sample, mean, log_var, log_prob, entropy

        # model = torch.nn.Sequential(
        #     self.fc1,
        #     nn.Dropout(p=0.6),
        #     nn.ReLU(),
        #     self.l2,
        #     nn.Softmax(dim=-1)
        # )
        # return model(x)