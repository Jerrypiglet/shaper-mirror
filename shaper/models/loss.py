import torch
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
        cls_logit = preds["cls_logit"]
        cls_label = labels["cls_label"]
        if self.label_smoothing > 0:
            cls_loss = smooth_cross_entropy(cls_logit, cls_label, self.label_smoothing)
        else:
            cls_loss = F.cross_entropy(cls_logit, cls_label)
        loss_dict = {
            'cls_loss': cls_loss,
        }
        return loss_dict

class PartSegLoss(nn.Module):
    """Part segmentation loss"""

    def __init__(self):
        super(PartSegLoss, self).__init__()

    def forward(self, preds, labels):
        seg_logit = preds["seg_logit"]
        seg_label = labels["seg_label"]
        seg_loss = F.cross_entropy(seg_logit, seg_label)
        loss_dict = {
            "seg_loss": seg_loss
        }

        return loss_dict


class SemSegLoss(nn.Module):
    """Semantic segmentation loss, with weighted cross entropy"""

    def forward(self, preds, labels):
        seg_logit = preds["seg_logit"]
        seg_label = labels["seg_label"].long()
        seg_loss = F.cross_entropy(seg_logit, seg_label, reduction='none')
        if "label_weights" in labels:
            weights = labels["label_weights"]
            # Instance-wise weighting (similar to tensorflow's sparse_cross_entropy loss);
            # Pytorch's weighted cross entropy fxn ('weight' kwarg in F.cross_entropy) weights by class, not by instance.
            seg_loss = seg_loss.float() * weights.float()
        seg_loss = seg_loss.mean()
        loss_dict = {
            "seg_loss": seg_loss
        }
        return loss_dict
