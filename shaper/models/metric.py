from torch import nn


class ClsAccuracy(nn.Module):
    def forward(self, preds, labels):
        cls_logit = preds["cls_logit"]
        cls_label = labels["cls_label"]
        pred_label = cls_logit.argmax(1)

        cls_acc = pred_label.eq(cls_label).float()

        return {"cls_acc": cls_acc}


class SegAccuracy(nn.Module):
    def forward(self, preds, labels):
        seg_logit = preds["seg_logit"]
        seg_label = labels["seg_label"]
        pred_label = seg_logit.argmax(1)

        seg_acc = pred_label.eq(seg_label).float()

        return {"seg_acc": seg_acc}


class PartSegAccuracy(nn.Module):
    def __init__(self):
        super(PartSegAccuracy, self).__init__()
        self.cls_acc = ClsAccuracy()
        self.seg_acc = SegAccuracy()

    def forward(self, preds, labels):
        metric_dict = {}
        metric_dict.update(self.cls_acc.forward(preds, labels))
        metric_dict.update(self.seg_acc.forward(preds, labels))
        return metric_dict
