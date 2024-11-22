# https://github.com/manzilzaheer/DeepSets #
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only

def clip_grad(model, max_norm):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    clip_coef = max_norm / (total_norm + 1e-06)
    if clip_coef < 1:
        for p in model.parameters():
            p.grad.data.mul_(clip_coef)
    return total_norm

class PermEqui1_max(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(PermEqui1_max, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        xm, _ = x.max(1, keepdim=True)
        x = self.Gamma(x - xm)
        return x

class PermEqui1_mean(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(PermEqui1_mean, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        xm = x.mean(1, keepdim=True)
        x = self.Gamma(x - xm)
        return x

class PermEqui2_max(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(PermEqui2_max, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        xm, _ = x.max(1, keepdim=True)
        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm
        return x

class PermEqui2_mean(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(PermEqui2_mean, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        xm = x.mean(1, keepdim=True)
        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm
        return x

class D(nn.Module):

    def __init__(self, d_dim, x_dim=3, num_classes=40, pool='mean'):
        super(D, self).__init__()
        self.d_dim = d_dim
        self.x_dim = x_dim
        if pool == 'max':
            self.phi = nn.Sequential(PermEqui2_max(self.x_dim, self.d_dim), nn.ELU(inplace=True), PermEqui2_max(self.d_dim, self.d_dim), nn.ELU(inplace=True), PermEqui2_max(self.d_dim, self.d_dim), nn.ELU(inplace=True))
        elif pool == 'max1':
            self.phi = nn.Sequential(PermEqui1_max(self.x_dim, self.d_dim), nn.ELU(inplace=True), PermEqui1_max(self.d_dim, self.d_dim), nn.ELU(inplace=True), PermEqui1_max(self.d_dim, self.d_dim), nn.ELU(inplace=True))
        elif pool == 'mean':
            self.phi = nn.Sequential(PermEqui2_mean(self.x_dim, self.d_dim), nn.ELU(inplace=True), PermEqui2_mean(self.d_dim, self.d_dim), nn.ELU(inplace=True), PermEqui2_mean(self.d_dim, self.d_dim), nn.ELU(inplace=True))
        elif pool == 'mean1':
            self.phi = nn.Sequential(PermEqui1_mean(self.x_dim, self.d_dim), nn.ELU(inplace=True), PermEqui1_mean(self.d_dim, self.d_dim), nn.ELU(inplace=True), PermEqui1_mean(self.d_dim, self.d_dim), nn.ELU(inplace=True))
        self.ro = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(self.d_dim, self.d_dim), nn.ELU(inplace=True), nn.Dropout(p=0.5), nn.Linear(self.d_dim, num_classes))
        print(self)

    def forward(self, x):
        phi_output = self.phi(x)
        sum_output = phi_output.mean(1)
        ro_output = self.ro(sum_output)
        return ro_output

class DTanh(nn.Module):

    def __init__(self, d_dim, x_dim=3, pool='mean'):
        super(DTanh, self).__init__()
        self.d_dim = d_dim
        self.x_dim = x_dim
        if pool == 'max':
            self.phi = nn.Sequential(PermEqui2_max(self.x_dim, self.d_dim), nn.Tanh(), PermEqui2_max(self.d_dim, self.d_dim), nn.Tanh(), PermEqui2_max(self.d_dim, self.d_dim), nn.Tanh())
        elif pool == 'max1':
            self.phi = nn.Sequential(PermEqui1_max(self.x_dim, self.d_dim), nn.Tanh(), PermEqui1_max(self.d_dim, self.d_dim), nn.Tanh(), PermEqui1_max(self.d_dim, self.d_dim), nn.Tanh())
        elif pool == 'mean':
            self.phi = nn.Sequential(PermEqui2_mean(self.x_dim, self.d_dim), nn.Tanh(), PermEqui2_mean(self.d_dim, self.d_dim), nn.Tanh(), PermEqui2_mean(self.d_dim, self.d_dim), nn.Tanh())
        elif pool == 'mean1':
            self.phi = nn.Sequential(PermEqui1_mean(self.x_dim, self.d_dim), nn.Tanh(), PermEqui1_mean(self.d_dim, self.d_dim), nn.Tanh(), PermEqui1_mean(self.d_dim, self.d_dim), nn.Tanh())
        self.ro = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(self.d_dim, self.d_dim), nn.Tanh(), nn.Dropout(p=0.5), nn.Linear(self.d_dim, 40))
        print(self)

    def forward(self, x):
        phi_output = self.phi(x)
        sum_output, _ = phi_output.max(1)
        ro_output = self.ro(sum_output)
        return ro_output

class DeepSetsModel(nn.Module):

    def __init__(self, dim_input, num_classes, d_dim=128, pool='mean'):
        super(DeepSetsModel, self).__init__()
        self.model = D(d_dim=d_dim, x_dim=dim_input, num_classes=num_classes, pool=pool)
        self.num_classes = num_classes

    def forward(self, x):
        logits = self.model(x)
        probabilities = F.softmax(logits, dim=-1)
        return (logits, probabilities)