import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, act):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            eval(act),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            eval(act),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            eval(act),
            nn.Linear(hidden_dim[2], hidden_dim[3]),
            eval(act),
            nn.Linear(hidden_dim[3], hidden_dim[4])
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim[4], hidden_dim[3]),
            eval(act),
            nn.Linear(hidden_dim[3], hidden_dim[2]),
            eval(act),
            nn.Linear(hidden_dim[2], hidden_dim[1]),
            eval(act),
            nn.Linear(hidden_dim[1], hidden_dim[0]),
            eval(act),
            nn.Linear(hidden_dim[0], input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return latent, decoded

    def get_intermediate_layers(self, x):
        intermediate_outputs = []
        x = x
        for layer in self.encoder:
            x = layer(x)
            intermediate_outputs.append(x)
        return intermediate_outputs