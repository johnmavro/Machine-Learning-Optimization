import torch


def projection(x, p, r):
    """
    Computes the solution of the oracle for the L^p norm ball of radius r
    Note: set p = "infty" for the L infinity norm
    :param x: point to be projected
    :param p: exponent of the p-norm
    :param r: radius of the ball
    :return: the projection of x
    """
    if torch.norm(x, p) > r:
        if str(p) > str(1) and str(p) != "infty":
            q = 1 / (1 - 1/p)
            return - r * torch.sign(x) * torch.abs(x) ** (q/p) / torch.norm(x, p) ** (q/p)
        elif str(p) == str(1):
            best_augment = torch.zeros(x.size(dim=0))
            best_augment[torch.argmax(torch.abs(x))] = 1
            return - r * torch.sign(x) * best_augment
        elif str(p) == "infty":
            return - r * torch.sign(x)
    else:
        return x


def K_sparse_polytope(x, K, r):
    """
    Computes the oracle for the K sparse polytope which is the intersection of the L^1 ball of radius r
    and the L^ infty ball of radius K*r
    :param x: point for the oracle
    :param K: polytope parameter
    :param r: radius of the polytope
    :return: solution of the oracle
    """
    assert K <= x.size(dim=0)
    best_augment = torch.argsort(x)[:-K]
    return torch.tensor(-r * torch.sign(x[i]) if i in best_augment else 0 for i in range(x.size(dim=0)))


def K_norm_ball():
    raise NotImplementedError
