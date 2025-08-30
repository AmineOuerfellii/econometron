import torch.nn as nn
import torch.nn.functional as F
import torch

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
    def forward(self, x, mode: str):
        if mode == "norm":
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        else:
            raise ValueError("Mode should be either 'norm' or 'denorm'")
    def _normalize(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + self.eps
        x_norm = (x - mean) / std
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta
        return x_norm, (mean.detach(), std.detach())
    def _denormalize(self, x_tuple):
        x_norm, (mean, std) = x_tuple 
        if self.affine:
            x_norm = (x_norm - self.beta) / (self.gamma + self.eps)
        x_denorm = x_norm * std + mean
        return x_denorm
