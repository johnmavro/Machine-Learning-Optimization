import torch.optim as optim
from Frank_Wolfe.MultiClassHingeLoss import *
from collections import defaultdict
from utils.utils import *


class DFW(optim.Optimizer):
    """
    Class implementing a Deep Frank-Wolfe Optimizer
    For a reference, please see
    "Deep Frank-Wolfe for Neural Network Optimization" https://arxiv.org/pdf/1811.07591.pdf
    For a reference about the multistep algorithm, please see the report provided in the GitHub repository.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        eta (float): proximal coefficient
        momentum (float, optional): momentum factor (default: 0.9)
            - Note: adding the momentum proved empirically to increase the performance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): small constant for numerical stability (default: 1e-5)
        name(string, optional): a string which recognizes the optimizer, because the optimizer.step() needs to have
            access to the loss, which is not the case for other state-of-the-art optimizers (default: "DFW")
        lambda_(float, optional): in the standard algorithm this is always zero (default: 0)
            - Note: because we perform multiple proximal steps this has to be updated as self.gamma
        tol (float, optional): we tried a criterion to pass from the multistep to the single step algorithm of the form
            self.lambda_ / norm(s_t) <= tol for some user-defined tolerance tol (default: 1e-2), s_t the dual direction
        prox_steps(int, optional): number of steps of Proximal Frank-Wolfe Algorithm to be performed (default: 2)
            - Note: in the experiments we always perform an additional step to retain a good accuracy while containing
                    the computational cost. For more details, please refer to the report on the GitHub repository.
    """

    def __init__(self, params, eta=1e-3, momentum=0.9, weight_decay=0, eps=1e-5, name="DFW",
                 lambda_=0, tol=1e-2, prox_steps=2):

        # check on the proximal coefficient
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
        """
        Optimizer step method
        :param closure: to have access to the loss
        :param model: current model (needs to be evaluated for the multistep case)
        :param batch_x: batch input (used only for the multistep case)
        :param batch_y: batch target (used only for the multistep case)
        :return:
        """
        loss = float(closure())  # compute loss

        w_dict = defaultdict(dict)  # initialization of w_dict
        for group in self.param_groups:
            wd = group['weight_decay']  # get weight decay parameter (we always set this parameter to zero)
            for param in group['params']:
                if param.grad is None:
                    continue
                w_dict[param]['delta_t'] = param.grad.data  # gradient of the objective
                w_dict[param]['r_t'] = wd * param.data  # gradient of the regularization term

        criterion = MultiClassHingeLoss()  # needs to be convex and piecewise linear

        w_0_dict = self._line_search(loss, w_dict)  # line search for the optimal self.gamma on the single step

        for group in self.param_groups:
            eta = group['eta']
            mu = group['momentum']
            for param in group['params']:
                if param.grad is None:
                    continue
                state = self.state[param]
                # get conditional gradients
                delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']

                # update weights
                param.data -= eta * (r_t + self.gamma * delta_t)

                # momentum if present
                if mu:
                    z_t = state['momentum_buffer']
                    z_t *= mu
                    z_t -= eta * self.gamma * (delta_t + r_t)
                    param.data += mu * z_t

        # we perform prox_steps - 1 additional steps of the proximal Frank-Wolfe Algorithm
        # for a reference, see Algorithm 2 in the provided report
        for i in range(self.prox_steps-1):

            # forward pass
            pred = model.train()(batch_x)
            loss = criterion(pred, batch_y)

            # proximal step to update self.gamma and self.lambda_
            self._proximal_step(loss, w_dict, w_0_dict)

            # update the network
            for group in self.param_groups:
                eta = group['eta']
                mu = group['momentum']
                for param in group['params']:
                    if param.grad is None:
                        continue
                    w_0 = w_0_dict[param]  # initialization weights
                    state = self.state[param]
                    delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']

                    # update weights
                    param.data *= (1 - self.gamma)
                    param.data += self.gamma * (-eta * (delta_t + r_t) + w_0)

                    # momentum if present
                    if mu:
                        z_t = state['momentum_buffer']
                        z_t *= mu
                        z_t -= eta * self.gamma * (delta_t + r_t)
                        param.data += mu * z_t

    @torch.autograd.no_grad()
    def _line_search(self, loss, w_dict):
        """
        Computes the line search in closed form, i.e. the optimal step-size.
        :param loss: value of the current loss
        :param w_dict: conditional gradients' dictionary
        :return:
            - w_0_dict: dictionary of all the initial weights, needed in case multiple proximal steps are performed
        """
        # set numerator and denominator
        num = loss
        denom = 0

        # solve for the first proximal step
        w_0_dict = {}  # save initial weights values in case multiple proximal steps are performed (prox_steps > 1)
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
        self.lambda_ = self.gamma * loss  # update also self.lambda_ in case prox_steps > 1

        return w_0_dict

    @torch.autograd.no_grad()
    def _proximal_step(self, loss, w_dict, w_0_dict):
        """
        Computes a non-trivial proximal step
        For a reference, see the report provided in the GitHub repository
        :param loss: loss at the current iterate
        :param w_dict: conditional gradients' dictionary
        :param w_0_dict: initial weights' dictionary
        :return:
        """
        # set numerator and denominator
        num = loss
        denom = 0

        for group in self.param_groups:
            wd = group['weight_decay']  # weight decay
            eta = group['eta']  # proximal coefficient
            for param in group['params']:
                if param.grad is None:
                    continue
                w_0 = w_0_dict[param]  # from the first proximal step
                w_dict[param]['delta_t'] = param.grad.data  # gradient of the objective
                w_dict[param]['r_t'] = wd * param.data  # gradient of the regularization

                delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']

                # update of numerator and numerator, contribution of the current layer
                num += 1/eta * torch.sum((param.data + eta * (delta_t + r_t) - w_0) * (param.data - w_0))
                denom += 1/eta * torch.norm(param.data + eta * (delta_t + r_t) - w_0) ** 2

        num -= self.lambda_  # self.lambda_ is not always zero in the multistep case
        self.gamma = float((num / (denom + self.eps)).clamp(min=0, max=1))  # update self.gamma
        self.lambda_ *= (1 - self.gamma)  # update self.lambda_
        self.lambda_ += self.gamma * loss  # update self.lambda_
