import torch.nn as nn
import pyro.distributions as dist
import torch
import pyro

class Encoder(nn.Module):

    def __init__(self, z_dim, hidden_dim):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)

        self.softplus = nn.Softplus()

    def forward(self, x):

        x = x.reshape(-1, 784)

        hidden = self.softplus(self.fc1(x))
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))

        return z_loc, z_scale

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
