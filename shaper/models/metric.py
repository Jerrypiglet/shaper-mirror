from torch import nn


class Accuracy(nn.Module):
    def forward(self, preds, labels):
        cls_logit = preds["cls_logit"]
        cls_label = labels["cls_label"]
        pred_label = cls_logit.argmax(1)

        acc = pred_label.eq(cls_label).float()

        return {"acc": acc}
