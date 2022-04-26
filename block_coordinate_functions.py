import torch
import torch.nn as nn



def update_v_js(U1, U2, W, b, rho, gamma):
    """
    The function updates the V_js parameters during the training phase
    :param U1: The U parameter on the same level of V that we are updating
    :param U2: The U parameter which is in the next level of the V that we are updating
    :param W: The W parameter which is in the next level of the V that we are updating
    :param b: The b parameter which is in the next level of the V that we are updating
    :param rho: The constant rho parameter which is in the next level of the V that we are updating
    :param gamma: The constant gamma parameter which is in the next level of the V that we are updating
    :return: The updated V
    """
    _, d = W.size()
    I = torch.eye(d, device=device)
    U1 = nn.ReLU()(U1)
    _, col_U2 = U2.size()
    Vstar = torch.mm(torch.inverse(rho * (torch.mm(torch.t(W), W)) + gamma * I),
                     rho * torch.mm(torch.t(W), U2 - b.repeat(1, col_U2)) + gamma * U1)
    return Vstar


def update_wb_js(U, V, W, b, alpha, rho):
    """
    The function updates the W and b parameters during the training phase
    :param U: The U in the current level of W and b
    :param V: The V in the previous level with respect to the W that we are updating
    :param W: The current W that we have to update
    :param b: The current b that we have to update
    :param alpha: The alpha constant of the updates
    :param rho: The rho constant of the updates
    :return:
    """
    d, N = V.size()
    I = torch.eye(d, device=device)
    _, col_U = U.size()
    Wstar = torch.mm(alpha * W + rho * torch.mm(U - b.repeat(1, col_U), torch.t(V)),
                     torch.inverse(alpha * I + rho * (torch.mm(V, torch.t(V)))))
    bstar = (alpha * b + rho * torch.sum(U - torch.mm(W, V), dim=1).reshape(b.size())) / (rho * N + alpha)
    return Wstar, bstar


def relu_prox(a, b, gamma, d, N):
    """
    The
    :param a: the a in the closed formula of the linearized update
    :param b: the b in the closed formula of the linearized update
    :param gamma: The constant used in the update
    :param d: the dimension of the current layer
    :param N: The number of samples
    :return: The obtained solution of the prox update
    """
    val = torch.empty(d, N, device=device)
    x = (a + gamma * b) / (1 + gamma)
    y = torch.min(b, torch.zeros(d, N, device=device))
    # torch.zeros(d,N, device=device)
    val = torch.where(a + gamma * b < 0, y, torch.zeros(d, N, device=device))
    val = torch.where(
        ((a + gamma * b >= 0) & (b >= 0)) | ((a * (gamma - np.sqrt(gamma * (gamma + 1))) <= gamma * b) & (b < 0)), x,
        val)
    val = torch.where((-a <= gamma * b) & (gamma * b <= a * (gamma - np.sqrt(gamma * (gamma + 1)))), b, val)
    return val
