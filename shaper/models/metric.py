"""Metric

The metric_fn could be implemented as a nn.Module or an object.
When a model is trained or evaluated, a metric_fn will be called after each batch.
The metric_fn should implement __call__, which behaves as a function.
It is also required to have two methods, "train" and "eval".

Examples:
    class EvalMetric(object):
        def __call__(self, preds, labels)
        def train(self)
        def eval(self)

"""

import torch
from torch import nn


class MetricList(nn.ModuleList):
    def forward(self, *args, **kwargs):
        metrics = {}
        for metric in self:
            metrics.update(metric(*args, **kwargs))
        return metrics


class ClsAccuracy(nn.Module):
    def forward(self, preds, labels):
        cls_logit = preds["cls_logit"]
        cls_label = labels["cls_label"]
        pred_label = cls_logit.argmax(1)

        cls_acc = pred_label.eq(cls_label).float()

        return {"cls_acc": cls_acc}


class SegAccuracy(nn.Module):
    def __init__(self, reduction="mean"):
        """Segmentation accuracy

        Args:
            reduction (str): specifies the reduction to apply to the output
                - "none": no reduction will be applied
                - "mean": accuracy per instance

        """
        super(SegAccuracy, self).__init__()
        self.reduction = reduction

    def forward(self, preds, labels):
        seg_logit = preds["seg_logit"]
        seg_label = labels["seg_label"]
        pred_label = seg_logit.argmax(1)

        # (batch_size, num_points)
        seg_acc = pred_label.eq(seg_label).float()
        if self.reduction == "mean":
            seg_acc = seg_acc.mean(1)

        return {"seg_acc": seg_acc}


class PartSegMetric(nn.Module):
    def __init__(self, num_seg_classes):
        super(PartSegMetric, self).__init__()
        self.seg_acc = SegAccuracy()
        # Disable IOU metric since it is inconsistent with ShapeNet IOU metric.
        # self.seg_iou = IntersectionAndUnion(num_seg_classes)

    def forward(self, preds, labels):
        metrics = self.seg_acc(preds, labels)
        # if not self.training:
        #     metrics.update(self.seg_iou(preds, labels))
        return metrics


class SemSegMetric(nn.Module):
    def __init__(self, num_seg_classes):
        super().__init__()
        self.seg_acc = SegAccuracy()

    def forward(self, preds, labels):
        metrics = self.seg_acc(preds, labels)
        return metrics


class IntersectionAndUnion(nn.Module):
    """Intersection and union

    The IOU for semantic segmentation is computed as follows:
    IOU = TP / (TP + FP + FN), where we call TP, union, and TP + FP + FN, intersection

    We use a common trick to compute the intersection so that we do not have to compute TN.
    total number of predictions = TP + FP
    total number of labels      = TP + FN
    So:
    union = TP + FP + FN = total number of predictions + total number of labels - TP

    In this module, we compute the class-wise intersection and union over a batch.
    When reduction is "mean", it will give mean IOU over valid classes.

    References: https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/utils.py

    Notes:
        Shapenet part segmentation:
        - 16 object categories (airplane, chair, motorbike)
        - 50 part classes (each object category has 2-6 part classes)
    """

    def __init__(self, num_classes, reduction="mean"):
        super(IntersectionAndUnion, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, preds, labels):
        # (batch_size, num_seg_classes, num_points)
        seg_logit = preds["seg_logit"].cpu()
        seg_label = labels["seg_label"].cpu()
        pred_label = seg_logit.argmax(1)

        # intersection
        intersection = pred_label[(pred_label == seg_label)]
        # (num_seg_classes,)
        intersection = torch.histc(intersection.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)

        # union
        total_num_pred = torch.histc(pred_label.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)
        total_num_label = torch.histc(seg_label.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)
        union = total_num_pred + total_num_label - intersection

        iou = intersection.float() / torch.clamp(union, min=1).float()
        if self.reduction == "mean":
            iou = torch.masked_select(iou, union > 0).mean()
        return {"IOU": iou}
