import torch
import torch.optim as optim
import torch.nn as nn
from Frank_Wolfe.MultiClassHingeLoss import *

from collections import defaultdict


class DFW(optim.Optimizer):
    """
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        eta (float): initial learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): small constant for numerical stability (default: 1e-5)
    """

    def __init__(self, params, eta=1e-3, momentum=0.9, weight_decay=0, eps=1e-5, name="DFW",
                 lambda_=0, tol=1e2, prox_steps=2):

        # check on the learning rate
        if eta <= 0.0:
            raise ValueError("Invalid eta: {}".format(eta))

        # check on the momentum
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        # check on the weight decay
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        # setting parameters
        self.eps = eps
        self.name = name
        self.prox_steps = prox_steps

        self.lambda_ = lambda_
        self.tol = tol

        # defaults dictionary initialization
        defaults = dict(eta=eta, momentum=momentum, weight_decay=weight_decay)
        super(DFW, self).__init__(params, defaults)

        # setting momentum
        for group in self.param_groups:
            if group['momentum']:
                for p in group['params']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

    @torch.autograd.no_grad()
    def step(self, closure, model, batch_x, batch_y):
        loss = float(closure())

        w_dict = defaultdict(dict)
        for group in self.param_groups:
            wd = group['weight_decay']
            for param in group['params']:
                if param.grad is None:
                    continue
                w_dict[param]['delta_t'] = param.grad.data
                w_dict[param]['r_t'] = wd * param.data

        criterion = MultiClassHingeLoss()

        w_0_dict = self._line_search(loss, w_dict)

        for group in self.param_groups:
            eta = group['eta']
            mu = group['momentum']
            for param in group['params']:
                if param.grad is None:
                    continue
                state = self.state[param]
                delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']

                # update weights
                param.data -= eta * (r_t + self.gamma * delta_t)

                # momentum if present
                if mu:
                    z_t = state['momentum_buffer']
                    z_t *= mu
                    z_t -= eta * self.gamma * (delta_t + r_t)
                    param.data += mu * z_t

        for i in range(self.prox_steps-1):

            self._proximal_step(loss, w_dict, w_0_dict, criterion, model, batch_x, batch_y)

            # update the network
            for group in self.param_groups:
                eta = group['eta']
                mu = group['momentum']
                for param in group['params']:
                    if param.grad is None:
                        continue
                    w_0 = w_0_dict[param]
                    state = self.state[param]
                    delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']

                    # update weights
                    param.data *= (1 - self.gamma)
                    param.data += self.gamma * (eta * (delta_t + r_t) + w_0)

                    # momentum if present
                    if mu:
                        z_t = state['momentum_buffer']
                        z_t *= mu
                        z_t -= eta * self.gamma * (delta_t + r_t)
                        param.data += mu * z_t

    @torch.autograd.no_grad()
    def _line_search(self, loss, w_dict):
        """
        Computes the line search in closed form.
        """
        num = loss
        denom = 0

        # solve for the first proximal step
        w_0_dict = {}
        for group in self.param_groups:
            eta = group['eta']
            for param in group['params']:
                if param.grad is None:
                    continue
                delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']
                num -= eta * torch.sum(delta_t * r_t)
                denom += eta * delta_t.norm() ** 2
                w_0_dict[param] = param.data

        self.gamma = float((num / (denom + self.eps)).clamp(min=0, max=1))
        self.lambda_ = self.gamma * loss

        return w_0_dict

    @torch.autograd.no_grad()
    def _proximal_step(self, w_dict, w_0_dict, criterion, model, batch_x, batch_y):
        """
        Computes a non-trivial proximal step
        """

        batch_x, batch_y = batch_x.to(device="cuda:0"), batch_y.to(device="cuda:0")  # move to cuda if possible

        # at each iteration, calculate the prediction on a batch and set the numerator to be the current loss
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        num = loss
        denom = 0
        for group in self.param_groups:
            wd = group['weight_decay']
            eta = group['eta']
            for param in group['params']:
                if param.grad is None:
                    continue
                w_0 = w_0_dict[param]  # from the first proximal step
                w_dict[param]['delta_t'] = param.grad.data
                w_dict[param]['r_t'] = wd * param.data
                delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']
                # update of numerator and numerator, contribution of the current layer
                num += 1/eta * torch.sum((param.data + eta * (delta_t + r_t) - w_0) * (param.data - w_0))
                denom += 1/eta * torch.norm(param.data + eta * (delta_t + r_t) - w_0) ** 2

        num -= self.lambda_  # since lambda_ is not zero now
        self.gamma = float((num / (denom + self.eps)).clamp(min=0, max=1))  # update gamma
        # update lambda_
        self.lambda_ *= (1 - self.gamma)
        self.lambda_ += self.gamma * loss  # update lambda_
