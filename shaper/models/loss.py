import torch.nn as nn
import torch.nn.functional as F

from shaper.nn.functional import smooth_cross_entropy


class ClsLoss(nn.Module):
    """Classification loss with optional label smoothing

    Attributes:
        label_smoothing (float or 0): the parameter to smooth labels
    """

    def __init__(self, label_smoothing=0):
        super(ClsLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, preds, labels):
        cls_logits = preds["cls_logits"]
        cls_labels = labels["cls_labels"]
        if self.label_smoothing > 0:
            cls_loss = smooth_cross_entropy(cls_logits, cls_labels, self.label_smoothing)
        else:
            cls_loss = F.cross_entropy(cls_logits, cls_labels)
        loss_dict = {
            'cls_loss': cls_loss,
        }
        return loss_dict