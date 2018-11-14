import torch
import torch.nn.functional as F


def smooth_cross_entropy(input, target, label_smoothing):
    """Cross entropy with label smoothing

    Args:
        input (torch.Tensor): (N, C)
        target (torch.Tensor): (N,)
        label_smoothing (float):

    Returns:
        loss (torch.Tensor): scalar

    """
    assert input.dim() == 2 and target.dim() == 1
    assert isinstance(label_smoothing, float) and label_smoothing > 0.0
    batch_size, num_classes = input.shape
    one_hot = torch.zeros_like(input).scatter(1, target.view(-1, 1), 1)
    smooth_one_hot = one_hot * (1 - label_smoothing) + torch.ones_like(input) * (label_smoothing / num_classes)
    log_prob = F.log_softmax(input, dim=1)
    loss = F.kl_div(log_prob, smooth_one_hot, reduction="none").sum(1).mean()
    return loss
