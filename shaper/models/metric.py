from torch import nn


class Accuracy(nn.Module):
    def forward(self, preds, labels):
        cls_logits = preds["cls_logits"]
        cls_labels = labels["cls_labels"]
        pred_labels = cls_logits.argmax(1)

        acc = pred_labels.eq(cls_labels).float()

        return {"acc": acc}
