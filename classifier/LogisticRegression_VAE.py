import torch.nn as nn
import torch
import pyro.distributions as dist

class LogisticRegression(nn.Module):

    def __init__(self, number_hidden_units, num_classes, encoder):
        super(LogisticRegression, self).__init__()

        self.encoder = encoder
        self.linear = nn.Linear(number_hidden_units, num_classes, )

    def forward(self, x):

        z_loc, z_scale = self.encoder.forward(x)
        z = dist.Normal(z_loc, z_scale).sample()

        out = self.linear(z)

        return out