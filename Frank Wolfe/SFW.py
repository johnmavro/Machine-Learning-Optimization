from constraints.constraints import *


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
    def step(self, constraint_type, constraints, unconstrained=False, closure=None):

        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            constraints (callable, must be non-empty): Dictionary of constraints for solving the oracle
                        if one is solving the problem on p-norm balls then this dictionary should contain just p and r
            constraint_type: the type of constraint, a string which contains one of the functions in constraints.py
            unconstrained: if true there are no constraints and the oracle is not solved
        """

        if not unconstrained:

            # if the constraints are of projection onto L^p balls type
            if str(constraint_type) == "projection":
                if dict(constraints) is None:
                    raise ValueError("The Stochastic Frank Wolfe Algorithm works for constrained optimization, but no"
                                     "constraints were given as input")
                if "p" not in list(dict(constraints).keys()) or "r" not in list(dict(constraints).keys()):
                    raise ValueError("The parameters given for the SFW Algorithm are not correct, the dictionary "
                                     "should be of the form {p: p, r: r}")

            # if the constraints are of projection onto polytopes type
            elif str(constraint_type) == "K_sparse_polytope":
                if dict(constraints) is None:
                    raise ValueError("The Stochastic Frank Wolfe Algorithm works for constrained optimization, but no"
                                     "constraints were given as input")
                if "K" not in list(dict(constraints).keys()) or "r" not in list(dict(constraints).keys()):
                    raise ValueError("The parameters given for the SFW Algorithm are not correct, the dictionary "
                                     "should be of the form {K: k, r: r}")

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

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

                # solution of the projection oracle
                if not unconstrained:
                    constraints_keys = list(dict(constraints).keys())
                    d_p = constraint_type(d_p, constraints[constraints_keys[0]],
                                          constraints[constraints_keys[1]])

                # clamp the learning rate between 0 and 1
                lr = group['lr']

                # update of the current parameter
                p.mul_(1 - lr)
                p.add_(d_p, alpha=lr)

        return loss
