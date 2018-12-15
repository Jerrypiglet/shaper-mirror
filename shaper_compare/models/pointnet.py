import torch
import torch.nn as nn

from shaper.nn import MLP, SharedMLP
from shaper.nn.init import set_bn
from shaper.models.pointnet import Stem


class PointNetFeature(nn.Module):
    """PointNet Feature Extractor

    Structure: input -> [Stem] -> features -> [SharedMLP] -> local features
    -> [MaxPool] -> gloal features -> [MLP] -> output feature

    """

    def __init__(self,
                 in_channels,
                 stem_channels=(64, 64),
                 local_channels=(64, 128, 1024),
                 global_channels=(512, 256),
                 dropout_prob=0.5,
                 with_transform=True,
                 bn_momentum=0.1):
        super(PointNetFeature, self).__init__()

        self.in_channels = in_channels
        self.out_channels = global_channels[-1]

        self.stem = Stem(in_channels, stem_channels, with_transform=with_transform)
        self.mlp_local = SharedMLP(stem_channels[-1], local_channels)
        self.mlp_global = MLP(local_channels[-1], global_channels, dropout=dropout_prob)

        set_bn(self, momentum=bn_momentum)

    def forward(self, points):
        # stem
        x, end_points = self.stem(points)
        # mlp for local features
        x = self.mlp_local(x)
        # max pool over points
        x, max_indices = torch.max(x, 2)
        end_points['key_point_inds'] = max_indices
        # mlp for global features
        feature = self.mlp_global(x)

        return feature


def build_pointnet_feature(cfg):
    net = PointNetFeature(
        in_channels=cfg.INPUT.IN_CHANNELS,
        stem_channels=cfg.MODEL.POINTNET.STEM_CHANNELS,
        local_channels=cfg.MODEL.POINTNET.LOCAL_CHANNELS,
        global_channels=cfg.MODEL.POINTNET.GLOBAL_CHANNELS,
        dropout_prob=cfg.MODEL.POINTNET.DROPOUT_PROB,
        with_transform=cfg.MODEL.POINTNET.WITH_TRANSFORM,
    )

    return net
