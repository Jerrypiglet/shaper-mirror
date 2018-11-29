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
