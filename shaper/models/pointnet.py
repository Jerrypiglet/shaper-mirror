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

from ._utils import Conv1dBlock, LinearBlock


class PointNetLocal(nn.ModuleList):
    def __init__(self, in_channels,
                 local_channels=(64, 128, 1024),
                 bn=True):
        super(PointNetLocal, self).__init__()

        self.in_channels = in_channels

        for ind, out_channels in enumerate(local_channels):
            self.append(Conv1dBlock(in_channels, out_channels, relu=True, bn=bn))
            in_channels = out_channels

        self.out_channels = in_channels

    def forward(self, x):
        for module in self:
            x = module(x)
        return x


class PointNetGlobal(nn.ModuleList):
    def __init__(self, in_channels,
                 global_channels=(512, 256),
                 bn=True):
        super(PointNetGlobal, self).__init__()

        self.in_channels = in_channels

        for ind, out_channels in enumerate(global_channels):
            self.append(LinearBlock(in_channels, out_channels, relu=True, bn=bn))
            in_channels = out_channels

        self.out_channels = in_channels

    def forward(self, x):
        for module in self:
            x = module(x)
        return x


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
        self.mlp_local = PointNetLocal(in_channels, local_channels, bn=bn)

        # global features
        self.mlp_global = PointNetGlobal(self.mlp_local.out_channels, global_channels, bn=bn)

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


class PointNetStem(nn.Module):
    def __init__(self, in_channels,
                 stem_channels=(64, 64),
                 bn=True):
        super(PointNetStem, self).__init__()

        self.in_channels = in_channels

        # input transform
        self.transform_input = TNet(in_channels, in_channels, bn=bn)

        # feature stem
        self.stem = PointNetLocal(in_channels, stem_channels, bn=bn)
        self.out_channels = self.stem.out_channels

        # feature transform
        self.transform_stem = TNet(self.out_channels, self.out_channels, bn=bn)

    def forward(self, x):
        end_points = {}

        # input transform
        trans_input = self.transform_input(x)
        x = torch.bmm(trans_input, x)
        end_points['trans_input'] = trans_input

        # feature stem
        x = self.stem(x)

        trans_stem = self.transform_stem(x)
        x = torch.bmm(trans_stem, x)
        end_points['trans_stem'] = trans_stem

        return x, end_points


class PointNetCls(nn.Module):
    def __init__(self, in_channels, out_channels,
                 stem_channels=(64, 64),
                 local_channels=(64, 128, 1024),
                 global_channels=(512, 256),
                 bn=True):
        super(PointNetCls, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stem = PointNetStem(in_channels, stem_channels, bn=bn)
        self.mlp_local = PointNetLocal(self.stem.out_channels, local_channels, bn=bn)
        self.mlp_global = PointNetGlobal(self.mlp_local.out_channels, global_channels, bn=bn)
        self.linear = nn.Linear(self.mlp_global.out_channels, out_channels, bias=False)

        self.init_weights()

    def forward(self, data_batch):
        x = data_batch["points"]
        x, end_points = self.stem(x)

        x = self.mlp_local(x)
        x, max_indices = torch.max(x, 2)
        end_points['key_points'] = max_indices
        x = self.mlp_global(x)
        x = self.linear(x)

        preds = {
            'cls_logits': x
        }
        preds.update(end_points)

        return preds

    def init_weights(self):
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='linear')


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
            reg_loss = F.mse_loss(trans_norm, I)
            loss_dict["reg_loss"] = reg_loss

        return loss_dict


class PointNetMetric(nn.Module):
    def forward(self, preds, labels):
        cls_logits = preds["cls_logits"]
        cls_labels = labels["cls_labels"]
        pred_labels = cls_logits.argmax(1)
        acc = pred_labels.eq(cls_labels).float().mean()

        metric_dict = {
            'acc': acc
        }
        return metric_dict


def build_pointnet(cfg):
    if cfg.TASK == "classification":
        net = PointNetCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            stem_channels=cfg.MODEL.POINTNET.STEM_CHANNELS,
            local_channels=cfg.MODEL.POINTNET.LOCAL_CHANNELS,
            global_channels=cfg.MODEL.POINTNET.GLOBAL_CHANNELS,
        )
        loss_fn = PointNetClsLoss(cfg.MODEL.POINTNET.REG_WEIGHT)
        metric_fn = PointNetMetric()
    else:
        raise NotImplementedError()

    return net, loss_fn, metric_fn


if __name__ == '__main__':
    batch_size = 32
    in_channels = 3
    num_points = 2048
    num_classes = 10

    data = torch.rand(batch_size, in_channels, num_points)
    transform = TNet()
    out = transform(data)
    print('TNet', out.shape)

    pointnet = PointNetCls(in_channels, num_classes)
    out, _ = pointnet(data)
    print('pointnet', out.shape)
