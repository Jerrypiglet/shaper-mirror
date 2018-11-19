"""PointNet

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
from shaper.nn.init import set_bn
from shaper.models.metric import Accuracy


class TNet(nn.Module):
    """Transformation Network. The structure is similar with PointNet"""

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 local_channels=(64, 128, 1024),
                 global_channels=(512, 256)):
        super(TNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # local features
        self.mlp_local = SharedMLP(in_channels, local_channels)

        # global features
        self.mlp_global = MLP(local_channels[-1], global_channels)

        # linear output
        self.linear = nn.Linear(global_channels[-1], in_channels * out_channels, bias=True)

        self.init_weights()

    def forward(self, x):
        """TNet forward

        Args:
            x (torch.Tensor): (batch_size, in_channels, num_points)

        Returns:
            torch.Tensor: (batch_size, out_channels, in_channels)

        """
        x = self.mlp_local(x)  # (batch_size, local_channels[-1], num_points)
        x, _ = torch.max(x, 2)  # (batch_size, local_channels[-1])
        x = self.mlp_global(x)
        x = self.linear(x)
        x = x.view(-1, self.out_channels, self.in_channels)
        I = torch.eye(self.out_channels, self.in_channels, dtype=x.dtype, device=x.device)
        x = x.add(I)  # broadcast add
        return x

    def init_weights(self):
        # Initialize linear transform to 0
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


class Stem(nn.Module):
    """Stem (main body or stalk). Extract features from raw point clouds

    Structure: input (-> [TNet] -> transform_input) -> [MLP] -> features (-> [TNet] -> transform_feature)
    
    Attributes:
        with_transform: whether to use TNet

    """

    def __init__(self, in_channels,
                 stem_channels=(64, 64),
                 with_transform=True):
        super(Stem, self).__init__()

        self.in_channels = in_channels
        self.out_channels = stem_channels[-1]
        self.with_transform = with_transform

        # feature stem
        self.mlp = SharedMLP(in_channels, stem_channels)

        if self.with_transform:
            # input transform
            self.transform_input = TNet(in_channels, in_channels)
            # feature transform
            self.transform_feature = TNet(self.out_channels, self.out_channels)

    def forward(self, x):
        """PointNet Stem forward

        Args:
            x (torch.Tensor): (batch_size, in_channels, num_points)

        Returns:
            torch.Tensor: (batch_size, stem_channels[-1], num_points)
            dict (optional non-empty):
                trans_input: (batch_size, in_channels, in_channels)
                trans_feature: (batch_size, stem_channels[-1], stem_channels[-1])

        """
        end_points = {}

        # input transform
        if self.with_transform:
            trans_input = self.transform_input(x)
            x = torch.bmm(trans_input, x)
            end_points['trans_input'] = trans_input

        # feature
        x = self.mlp(x)

        # feature transform
        if self.with_transform:
            trans_feature = self.transform_feature(x)
            x = torch.bmm(trans_feature, x)
            end_points['trans_feature'] = trans_feature

        return x, end_points


# -----------------------------------------------------------------------------
# PointNet for classification
# -----------------------------------------------------------------------------
class PointNetCls(nn.Module):
    """PointNet for classification

    Structure: input -> [Stem] -> features -> [SharedMLP] -> local features
    -> [MaxPool] -> gloal features -> [MLP] -> [Linear] -> logits

    """

    def __init__(self,
                 in_channels, out_channels,
                 stem_channels=(64, 64),
                 local_channels=(64, 128, 1024),
                 global_channels=(512, 256),
                 dropout_prob=0.5,
                 with_transform=True):
        super(PointNetCls, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stem = Stem(in_channels, stem_channels, with_transform=with_transform)
        self.mlp_local = SharedMLP(stem_channels[-1], local_channels)
        self.mlp_global = MLP(local_channels[-1], global_channels, dropout=dropout_prob)
        self.linear = nn.Linear(global_channels[-1], out_channels, bias=True)

        self.init_weights()
        set_bn(self, momentum=0.01)

    def forward(self, data_batch):
        x = data_batch["points"]

        # stem
        x, end_points = self.stem(x)
        # mlp for local features
        x = self.mlp_local(x)
        # max pool over points
        x, max_indices = torch.max(x, 2)
        end_points['key_point_inds'] = max_indices
        # mlp for global features
        x = self.mlp_global(x)
        x = self.linear(x)

        preds = {
            'cls_logits': x
        }
        preds.update(end_points)

        return preds

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


class PointNetClsLoss(nn.Module):
    """Pointnet classification loss with optional regularization loss

    Attributes:
        reg_weight (float): regularization weight for feature transform matrix

    """

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
            trans_feature = preds["trans_feature"]
            trans_norm = torch.bmm(trans_feature.transpose(2, 1), trans_feature)  # [in, in]
            I = torch.eye(trans_norm.size(2), dtype=trans_norm.dtype, device=trans_norm.device)
            # CAUTION: torch.expand
            reg_loss = F.mse_loss(trans_norm, I.unsqueeze(0).expand_as(trans_norm), reduction="sum")
            loss_dict["reg_loss"] = reg_loss * (0.5 * self.reg_weight / trans_norm.size(0))

        return loss_dict


def build_pointnet(cfg):
    if cfg.TASK == "classification":
        net = PointNetCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            stem_channels=cfg.MODEL.POINTNET.STEM_CHANNELS,
            local_channels=cfg.MODEL.POINTNET.LOCAL_CHANNELS,
            global_channels=cfg.MODEL.POINTNET.GLOBAL_CHANNELS,
            dropout_prob=cfg.MODEL.POINTNET.DROPOUT_PROB,
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
        print('PointNet:', k, v.shape)
