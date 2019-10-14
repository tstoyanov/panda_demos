import torch
from torch import nn, optim
from torch.nn import functional as F

class STATE_VAE(nn.Module):
    def __init__(self, latent_space_dim):
        super(STATE_VAE, self).__init__()
        
        self.fc1 = nn.Linear(600, 300)
        self.fc21 = nn.Linear(300, latent_space_dim)  # mu layer
        self.fc22 = nn.Linear(300, latent_space_dim)  # logvariance layer
        self.fc3 = nn.Linear(latent_space_dim, 300)
        self.fc4 = nn.Linear(300, 600)
        self.fc5 = nn.Linear(600, 600)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar, no_noise):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        if no_noise:
            return mu
        
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = torch.sigmoid(self.fc4(h3))
        # return h4
        return self.fc5(h4)

    def forward(self, x, no_noise):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, no_noise)
        return self.decode(z), mu, logvar