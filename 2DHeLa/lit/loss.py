import torch
import numpy as np

def dice_loss(pred, target):
    """Cacluate dice loss
    Parameters
    ----------
        pred:
            predictions from the model
        target:
            ground truth label
    """

    smooth = 1.
    p_flat = pred.contiguous().view(-1)
    t_flat = target.contiguous().view(-1)
    intersection = (p_flat * t_flat).sum()
    a_sum = torch.sum(p_flat * p_flat)
    b_sum = torch.sum(t_flat * t_flat)
    return 1. - ((2. * intersection + smooth) / (a_sum + b_sum + smooth) )


def MU_loss(pred, target):
    """Loss function for MU-Lux algorithm
        Parameters
        ----------
            pred:
                predictions from the model
            target:
                ground truth label
    """
    p_flat = pred.contiguous().view(-1)
    t_flat = target.contiguous().view(-1)
    weights = np.ones(p_flat.shape).astype(np.float64)


