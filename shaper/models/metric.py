import torch
from torch import nn


class ClsAccuracy(nn.Module):
    def forward(self, preds, labels):
        cls_logits = preds["cls_logits"]
        cls_labels = labels["cls_labels"]
        pred_labels = cls_logits.argmax(1)

        acc = pred_labels.eq(cls_labels).float()

        return {"acc": acc}


class SegAccuracy(nn.Module):
    def forward(self, preds, labels):
        seg_logits = preds["seg_logits"]
        seg_labels = labels["seg_labels"]
        pred_labels = seg_logits.argmax(1)

        acc = pred_labels.eq(seg_labels).float().mean(1)

        return {"acc": acc}


class IntersectionAndUnion(nn.Module):
    """
    Intersection and union computation inspired by:
    Inspired by https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/utils.py

    The IOU for semantic segmentation is computed as follows:
    IOU = TP / (TP + FP + FN), where we call TP, union, and TP + FP + FN, intersection

    We use a common trick to compute the intersection so that we do not have to compute TN.
    total number of predictions = TP + FP
    total number of labels      = TP + FN
    So:
    union = TP + FP + FN = total number of predictions + total number of labels - TP

    In this module, we only compute the class-wise intersection and union. The final IOU
    must be computed in the end over all the data.

    Shapenet part segmentation:
    - 16 object categories (airplane, chair, motorbike)
    - 50 part classes (each object category has 2-6 part classes)
    """
    def __init__(self, num_classes):
        super(IntersectionAndUnion, self).__init__()
        self.num_classes = num_classes

    def forward(self, preds, labels):
        seg_logits = preds["seg_logits"].cpu()
        seg_labels = labels["seg_labels"].cpu()
        pred_labels = seg_logits.argmax(1)

        # intersection
        intersection = pred_labels[(pred_labels == seg_labels)]
        intersection = torch.histc(intersection.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)

        # union
        total_num_pred = torch.histc(pred_labels.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)
        total_num_labels = torch.histc(seg_labels.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)
        union = total_num_pred + total_num_labels - intersection

        return intersection, union
