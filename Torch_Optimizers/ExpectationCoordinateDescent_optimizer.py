import numpy as np
import numpy.random as rand
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader

class ExpectationCoordinateDescent(optim.Optimizer):
    """
    Object of the optimizer class which performs the cyclic coordinate descent
    algorithm. Namely at every epoch a cycle over all the parameters of the
    neural network and performs an update for each parameter
    """
    def __init__(self, params, lr):
        # param_groups = list(params)
        # self.param_groups = param_groups

        # if len(param_groups) == 0:
        #    raise ValueError("optimizer got an empty parameters list")
        # if not isinstance(param_groups[0], dict):
        #    param_groups = [{'params': param_groups}]

        # for param_group in param_groups:
        #   self.add_param_group(param_group)

        defaults = dict(lr=lr)
        self.params = params

        super(CyclicCoordinateDescent, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure

        for group in self.param_groups:
                lr = group["lr"]
                for p in group["params"]:
                    if p.grad is not None:
                        #print("Got here!")
                        probas = p.grad *(torch.abs(nn.functional.normalize(p.grad,p=1,dim = 0))).cpu().apply_(lambda x: 1 if x>random.uniform(0, 1) else 0).to(device)
                        p.data.add_(-lr, probas)
                        state = self.state[p]
        return loss

class RevExpectationCoordinateDescent(optim.Optimizer):
    """
    Object of the optimizer class which performs the cyclic coordinate descent
    algorithm. Namely at every epoch a cycle over all the parameters of the
    neural network and performs an update for each parameter
    """
    def __init__(self, params, lr):
        # param_groups = list(params)
        # self.param_groups = param_groups

        # if len(param_groups) == 0:
        #    raise ValueError("optimizer got an empty parameters list")
        # if not isinstance(param_groups[0], dict):
        #    param_groups = [{'params': param_groups}]

        # for param_group in param_groups:
        #   self.add_param_group(param_group)

        defaults = dict(lr=lr)
        self.params = params

        super(CyclicCoordinateDescent, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure

        for group in self.param_groups:
                lr = group["lr"]
                for p in group["params"]:
                    if p.grad is not None:
                        #print("Got here!")
                        probas = p.grad *(torch.abs(nn.functional.normalize(p.grad,p=1,dim = 0))).cpu().apply_(lambda x: 1 if x<random.uniform(0, 1) else 0).to(device)
                        p.data.add_(-lr, probas)
                        state = self.state[p]
        return loss
