import torch
import torch.optim as optim
import torch.nn as nn

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

    def __init__(self, params, eta=1e-3, momentum=0, weight_decay=0, eps=1e-5, name="DFW", lambda_=0, tol=1e-2):

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
        
        # dual variable
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
        
        iters = 3
        criterion = nn.CrossEntropyLoss()
        self._line_search(loss, w_dict, criterion, iters, model, batch_x, batch_y)

        for group in self.param_groups:
            eta = group['eta']
            mu = group['momentum']
            for param in group['params']:
                if param.grad is None:
                    continue
                if mu:
                    z_t = state['momentum_buffer']
                    z_t *= mu
                    z_t -= eta * self.gamma * (delta_t + r_t)
                    param.data += mu * z_t

    @torch.autograd.no_grad()
    def _line_search(self, loss, w_dict, criterion, iters, model, batch_x, batch_y):
        """
        Computes the line search in closed form.
        """
        num = loss
        denom = 0
        w_0_dict = {}
        for group in self.param_groups:
            eta = group['eta']
            for param in group['params']:
                if param.grad is None:
                    continue
                #print(param.data.size(dim=0))
                delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']
                num -= eta * torch.sum(delta_t * r_t)
                denom += eta * delta_t.norm() ** 2
                w_0_dict[param] = param.data
            
        self.gamma = float((num / (denom + self.eps)).clamp(min=0, max=1))
        param.data -= eta * (r_t + self.gamma * delta_t)
        self.lambda_ = (self.gamma/eta)*loss
        
        prediction = model(batch_x)
        current_loss = criterion(prediction, batch_y)
        if abs(loss-current_loss) / abs(loss) <= self.tol: 
            for iteration in range(iters):
                print('******* Solving iteration', iteration, 'of the proximal *********')
                for group in self.param_groups:
                    eta = group['eta']
                    for param in group['params']:
                        if param.grad is None:
                            continue
                        w_0 = w_0_dict[param]
                        delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']
                        param.data -= eta * (r_t + self.gamma * delta_t)
                        w_s = - eta * (delta_t + r_t)
                        s_t = r_t + delta_t
                        pred = model(batch_x)
                        loss = criterion(pred, batch_y)
                        num = loss - eta * self.lambda_ + torch.sum((param.data - w_0 - w_s) * (param.data - w_0))
                        denom = torch.norm(param.data - w_s - w_0) ** 2  # TODO: READD WO IN THE UPDATE
                        self.gamma = float((num / (denom + self.eps)).clamp(min=0, max=1))
                        param.data *= (1-self.gamma)
                        param.data += self.gamma * (w_s + w_0)  # TODO: READ WO IN THE UPDATE
                        self.lambda_ *= (1-self.gamma)
                        self.lambda_ += (self.gamma/eta) * loss