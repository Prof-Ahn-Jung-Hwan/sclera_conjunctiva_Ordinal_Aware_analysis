import torch
import torch.nn as nn


class VariantionalEncoder(nn.Module):
    def __init__(self, dim, var_range=[-30.0, 20.0]):
        super().__init__()
        self.var_range = var_range
        self.dim = dim
        self.fc1 = nn.Linear(dim, dim)
        self.quant = nn.Linear(dim, dim * 2)

    def forward(self, z):
        z = self.fc1(z)
        mu, logvar = torch.chunk(self.quant(z), 2, dim=-1)

        if self.training:
            logvar = torch.clamp(logvar, min=self.var_range[0], max=self.var_range[1])
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        else:
            return mu
