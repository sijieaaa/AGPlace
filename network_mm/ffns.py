


import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint
import torch.nn.functional as F
import torch
import torch.nn as nn

from tools.options import parse_arguments
opt = parse_arguments()


class ODEFunc(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, t, x):
        output = self.func(x)
        return output


class SDEFunc(nn.Module):
    noise_type = 'diagonal'
    sde_type = 'ito'
    def __init__(self, mufunc, sigmafunc):
        super().__init__()
        self.mufunc = mufunc
        self.sigmafunc = sigmafunc
    def f(self, t, y):
        mu = self.mufunc(y)
        return mu
    def g(self, t, y):
        sigma = self.sigmafunc(y)
        return sigma


class CDEFunc(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, t, x):
        # x: [b,seq,c]
        output = self.func(x)
        output = output.view(output.shape[0],-1,x.shape[1]) # [b,hidc,inc]
        return output


def select_act(act):
    if act is None:
        out = nn.Identity()
    elif act == 'id':
        out = nn.Identity()
    elif act == 'relu':
        out = nn.ReLU()
    elif act == 'tanh':
        out = nn.Tanh()
    elif act == 'sigmoid':
        out = nn.Sigmoid()
    else:
        raise NotImplementedError
    return out


# ==== fc
class FC(nn.Module):
    def __init__(self, indim, outdim, act=None):
        super().__init__()
        self.fc = nn.Linear(indim, outdim)
        self.act = select_act(act)
    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        return x

class FCODE(nn.Module):
    def __init__(self, dim, act=None):
        super().__init__()
        self.func = ODEFunc(FC(dim,dim,act))
    def forward(self, x):
        t = torch.tensor([0, 1]).float().type_as(x)
        out = odeint(self.func, x, t, method=opt.odeint_method, options={'step_size': opt.odeint_size},
                     rtol=opt.tol, atol=opt.tol)
        out = out[-1]
        return out


