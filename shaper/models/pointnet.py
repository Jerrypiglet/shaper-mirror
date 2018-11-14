"""
PointNet

References:
    @article{qi2016pointnet,
      title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
      author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
      journal={arXiv preprint arXiv:1612.00593},
      year={2016}
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.nn import MLP, SharedMLP
from shaper.models.metric import Accuracy


class TNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 local_channels=(64, 128, 1024),
                 global_channels=(512, 256),
                 bn=True):
        super(TNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # local features
        self.mlp_local = SharedMLP(in_channels, local_channels, bn=bn)

        # global features
        self.mlp_global = MLP(self.mlp_local.out_channels, global_channels, bn=bn)

        # linear output
        self.linear = nn.Linear(self.mlp_global.out_channels, in_channels * out_channels, bias=True)

        self.init_weights()

    def forward(self, x):
        x = self.mlp_local(x)  # [N, C, W]
        x, _ = torch.max(x, 2)  # [N, C]
        x = self.mlp_global(x)
        x = self.linear(x)
        x = x.view(-1, self.out_channels, self.in_channels)
        I = torch.eye(self.out_channels, self.in_channels, dtype=x.dtype, device=x.device)
        x.add_(I)  # broadcast first dimension
        return x

    def init_weights(self):
        # set linear transform be 0
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


class Stem(nn.Module):
    def __init__(self, in_channels,
                 stem_channels=(64, 64),
                 with_transform=True,
                 bn=True):
        super(Stem, self).__init__()

        self.in_channels = in_channels
        self.out_channels = stem_channels[-1]
        self.with_transform = with_transform

        # feature stem
        self.stem = SharedMLP(in_channels, stem_channels, bn=bn)

        if self.with_transform:
            # input transform
            self.transform_input = TNet(in_channels, in_channels, bn=bn)
            # feature transform
            self.transform_stem = TNet(self.out_channels, self.out_channels, bn=bn)

    def forward(self, x):
        end_points = {}

        # input transform
        if self.with_transform:
            trans_input = self.transform_input(x)
            x = torch.bmm(trans_input, x)
            end_points['trans_input'] = trans_input

        # feature stem
        x = self.stem(x)

        # feature transform
        if self.with_transform:
            trans_stem = self.transform_stem(x)
            x = torch.bmm(trans_stem, x)
            end_points['trans_stem'] = trans_stem

        return x, end_points


# -----------------------------------------------------------------------------
# PointNet for classification
# -----------------------------------------------------------------------------
class PointNetCls(nn.Module):
    def __init__(self, in_channels, out_channels,
                 stem_channels=(64, 64),
                 local_channels=(64, 128, 1024),
                 global_channels=(512, 256),
                 dropout_ratio=0.5,
                 with_transform=True,
                 bn=True):
        super(PointNetCls, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stem = Stem(in_channels, stem_channels, with_transform=with_transform, bn=bn)
        self.mlp_local = SharedMLP(self.stem.out_channels, local_channels, bn=bn)
        self.mlp_global = MLP(self.mlp_local.out_channels, global_channels, bn=bn)
        self.dropout = nn.Dropout(p=dropout_ratio, inplace=True)
        self.linear = nn.Linear(self.mlp_global.out_channels, out_channels, bias=True)

        self.init_weights()

    def forward(self, data_batch):
        x = data_batch["points"]
        x, end_points = self.stem(x)

        x = self.mlp_local(x)
        x, max_indices = torch.max(x, 2)
        end_points['key_points'] = max_indices
        x = self.mlp_global(x)
        x = self.dropout(x)
        x = self.linear(x)

        preds = {
            'cls_logits': x
        }
        preds.update(end_points)

        return preds

    def init_weights(self):
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='linear')
        nn.init.zeros_(self.linear.bias)


class PointNetClsLoss(nn.Module):
    def __init__(self, reg_weight):
        super(PointNetClsLoss, self).__init__()
        self.reg_weight = reg_weight

    def forward(self, preds, labels):
        cls_logits = preds["cls_logits"]
        cls_labels = labels["cls_labels"]
        cls_loss = F.cross_entropy(cls_logits, cls_labels)
        loss_dict = {
            'cls_loss': cls_loss,
        }

        # regularization over transform matrix
        if self.reg_weight > 0.0:
            trans_stem = preds["trans_stem"]
            trans_norm = torch.bmm(trans_stem, trans_stem.transpose(2, 1))  # [out, out]
            I = torch.eye(trans_norm.size()[1], dtype=trans_norm.dtype, device=trans_norm.device)
            reg_loss = F.mse_loss(trans_norm, I.unsqueeze(0).repeat(trans_norm.size(0), 1, 1))
            loss_dict["reg_loss"] = reg_loss

        return loss_dict


def build_pointnet(cfg):
    if cfg.TASK == "classification":
        net = PointNetCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            stem_channels=cfg.MODEL.POINTNET.STEM_CHANNELS,
            local_channels=cfg.MODEL.POINTNET.LOCAL_CHANNELS,
            global_channels=cfg.MODEL.POINTNET.GLOBAL_CHANNELS,
            dropout_ratio=cfg.MODEL.POINTNET.DROPOUT_RATIO,
            with_transform=cfg.MODEL.POINTNET.WITH_TRANSFORM,
        )
        loss_fn = PointNetClsLoss(cfg.MODEL.POINTNET.REG_WEIGHT)
        metric_fn = Accuracy()
    else:
        raise NotImplementedError()

    return net, loss_fn, metric_fn


if __name__ == '__main__':
    batch_size = 32
    in_channels = 3
    num_points = 1024
    num_classes = 40

    data = torch.rand(batch_size, in_channels, num_points)
    transform = TNet()
    out = transform(data)
    print('TNet', out.shape)

    pointnet = PointNetCls(in_channels, num_classes)
    out_dict = pointnet({"points": data})
    for k, v in out_dict.items():
        print('pointnet:', k, v.shape)
