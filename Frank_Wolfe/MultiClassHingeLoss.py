import torch.nn as nn
import torch


class MultiClassHingeLoss(nn.Module):
    """
    Creates a criterion that optimizes a multi-class classification hinge loss (margin-based loss)
     between input `x` (a 2D mini-batch `Tensor`) and output y.
    """
    smooth = False

    def __init__(self):
        super(MultiClassHingeLoss, self).__init__()
        self.smooth = False
        self._range = None

    def forward(self, x, y):
        """
        Computes the loss given a sample (x, y)
        :param x: input
        :param y: target
        :return:
            - The loss
        """
        # augmented scores
        aug = self._augmented_scores(x, y)
        # xi
        xi = self._compute_xi(x, aug, y)

        # scalar product between the two (rescaled)
        loss = torch.sum(aug * xi) / x.size(0)
        return loss

    def _augmented_scores(self, s, y):
        """
        Computes the MultiClassHingeLoss given s and y
        For a reference, see equation (2) in
        "Deep Frank-Wolfe for Neural Network Optimization" https://arxiv.org/pdf/1811.07591.pdf
        Here, such a loss is re-written as in (11) in the same paper
        :param s: prediction scores
        :param y: target scores
        :return:
            - the multiClassHingeLoss recomputed as in (11) with the classification mismatch delta
        """
        if self._range is None:
            delattr(self, '_range')
            self.register_buffer('_range', torch.arange(s.size(1), device=s.device)[None, :])

        delta = torch.ne(y[:, None], self._range).detach().float()
        return s + delta - s.gather(1, y[:, None])

    @torch.autograd.no_grad()
    def _compute_xi(self, s, aug, y):
        """

        :param s: prediction scores
        :param aug: augmented scores
        :param y: target scores
        :return:
        """

        # find argmax of augmented scores
        _, y_star = torch.max(aug, 1)
        # xi_max: one-hot encoding of maximal indices
        xi_max = torch.eq(y_star[:, None], self._range).float()

        if MultiClassHingeLoss.smooth:
            # find smooth argmax of scores
            xi_smooth = nn.functional.softmax(s, dim=1)
            # compute for each sample whether it has a positive contribution to the loss
            losses = torch.sum(xi_smooth * aug, 1)
            mask_smooth = torch.ge(losses, 0).float()[:, None]
            # keep only smoothing for positive contributions
            xi = mask_smooth * xi_smooth + (1 - mask_smooth) * xi_max
        else:
            xi = xi_max

        return xi

    def __repr__(self):
        return 'MultiClassHingeLoss()'


class set_smoothing_enabled(object):
    def __init__(self, mode):
        """
        Context-manager similar to the torch.set_grad_enabled(mode).
        Within the scope of the manager the MultiClassHingeLoss is smoothed (it is not otherwise).
        The smoothing for the MultiClassHingeLoss is useful when dealing with a lot of classes (e.g. CIFAR100)
        """
        MultiClassHingeLoss.smooth = bool(mode)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        MultiClassHingeLoss.smooth = False
