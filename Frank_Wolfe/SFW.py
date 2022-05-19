from Frank_Wolfe.constraints.constraints import *


class SFW(torch.optim.Optimizer):
    """Stochastic Frank Wolfe Algorithm
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        learning_rate (float): learning rate between 0.0 and 1.0, should be decreasing to bound the variance
        momentum (float): momentum factor, 0 for no momentum
    """

    def __init__(self, params, learning_rate=0.1, rescale='diameter', momentum=0.9, name="SFW"):

        # check on the learning rate
        if not (0.0 <= learning_rate <= 1.0):
            raise ValueError("Invalid learning rate: {}".format(learning_rate))

        # check on the momentum
        if not (0.0 <= momentum <= 1.0):
            raise ValueError("Momentum must be between [0, 1].")

        # check on the rescaling type
        if not (rescale in ['diameter', 'gradient', None]):
            raise ValueError("Rescale type must be either 'diameter', 'gradient' or None.")

        # adding the parameters
        self.rescale = rescale
        self.name = name

        # initialize the dictionary
        defaults = dict(lr=learning_rate, momentum=momentum)

        # call the __init__
        super(SFW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, constraints, closure=None):

        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            constraints (callable, must be non-empty): Dictionary of constraints for solving the oracle
        """

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                # adding the momentum
                momentum = group['momentum']
                if momentum > 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = d_p.detach().clone()
                    else:
                        # update the momentum
                        param_state['momentum_buffer'].mul_(momentum).add_(d_p, alpha=1-momentum)
                        d_p = param_state['momentum_buffer']

                v = constraints[idx].lmo(d_p)  # LMO optimal solution

                if self.rescale == 'diameter':
                    # Rescale lr by diameter
                    factor = 1. / constraints[idx].get_diameter()
                elif self.rescale == 'gradient':
                    # Rescale lr by gradient
                    factor = torch.norm(d_p, p=2) / torch.norm(p - v, p=2)
                else:
                    # No rescaling
                    factor = 1

                lr = max(0.0, min(factor * group['lr'], 1.0))  # Clamp between [0, 1]

                # update of the current parameter
                p.mul_(1 - lr)
                p.add_(v, alpha=lr)
                idx += 1
        return loss
