from VariationalAutoEncoder.Encoder import Encoder
from VariationalAutoEncoder.Decoder import Decoder
import pyro.distributions as dist
import torch.nn as nn
import pyro
import torch

class VAE(nn.Module):

    def __init__(self, z_dim=30, hidden_dim=400, use_cuda=False):
        super(VAE, self).__init__()

        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            self.cuda()

        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x):

        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):

            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))

            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            loc_img = self.decoder.forward(z)
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):

        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):

            z_loc, z_scale = self.encoder.forward(x)
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):

        z_loc, z_scale = self.encoder(x)
        z = dist.Normal(z_loc, z_scale).sample()
        loc_img = self.decoder(z)

        return loc_img





