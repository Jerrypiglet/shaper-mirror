"""PointNet++

References:
    @article{qi2017pointnetplusplus,
      title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
      author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
      journal={arXiv preprint arXiv:1706.02413},
      year={2017}
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.nn import SharedMLP, Conv1d
from shaper.models.pointnet2.modules import PointNetSAModule, PointnetFPModule
from shaper.nn.init import set_bn

class PointNet2SSGPartSeg(nn.Module):
    """PointNet 2 part segmentation with single-scale grouping

    Structure: input -> [PointNetSA]s -> [Local Feature Extraction Layer] ->
                    [PointNetFP]s -> [FC layer]s

    PointNetSA: PointNet Set Abstraction Layer
    Local Feature Extraction Layer is a layer that converts the output of SAs to new features
    PointNetFP: PointNet Feature Propagation Layer
    """

    def __init__(self,
                 in_channels,
                 num_seg_classes,
                 num_centroids=(512, 128),
                 radius=(0.2, 0.4),
                 num_neighbours=(64, 64),
                 sa_channels=((64, 64, 128), (128, 128, 256)),
                 local_channels=(256, 512, 1024),
                 fp_channels=((256, 256), (256, 128), (128, 128, 128)),
                 num_fp_neighbour=(3, 3, 3),
                 seg_channels=(128,),
                 dropout=0.5,
                 use_xyz=True):
        """

        :param in_channels (int): the number of input channels
        :param num_seg_classes (int): the number of segmentation class
        :param num_centroids (tuple of int): the numbers of centroids to sample in each set abstraction module
        :param radius (tuple of float): a tuple of radius to query neighbours in each set abstraction module
        :param num_neighbours (tuple of int): the numbers of neighbours to query for each centroid
        :param sa_channels (tuple of tuple of int): the numbers of channels within each set abstraction module
        :param local_channels (tuple of int): the numbers of channels to extract local features after set abstraction
        :param fp_channels (tuple of tuple of int): the numbers of channels for feature propagation (FP) module
        :param num_fp_neighbour (tuple of int): the numbers of nearest neighbor used in FP
        :param seg_channels (tuple of int): the numbers of channels in segmentation mlp
        :param dropout (float): the probability to dropout input features
        :param use_xyz (bool): whether or not to use the xyz position of a points as a feature
        """
        super(PointNet2SSGPartSeg, self).__init__()

        self.in_channels = in_channels
        self.use_xyz = use_xyz

        # Sanity check
        num_sa_layers = len(num_centroids)
        num_fp_layers = len(fp_channels)
        assert len(radius) == num_sa_layers
        assert len(num_neighbours) == num_sa_layers
        assert len(sa_channels) == num_sa_layers
        assert (num_sa_layers + 1) == num_fp_layers     # SA layer + Local feature extraction layer = FP layers
        assert len(num_fp_neighbour) == num_fp_layers

        # Set Abstraction Layers
        feature_channels = in_channels - 3
        self.sa_modules = nn.ModuleList()
        for ind in range(num_sa_layers):
            sa_module = PointNetSAModule(in_channels=feature_channels,
                                         mlp_channels=sa_channels[ind],
                                         num_centroids=num_centroids[ind],
                                         radius=radius[ind],
                                         num_neighbours=num_neighbours[ind],
                                         use_xyz=use_xyz)
            self.sa_modules.append(sa_module)
            feature_channels = sa_channels[ind][-1]

        # Local Feature Extraction Layers
        if use_xyz:
            feature_channels += 3
        self.mlp_local = SharedMLP(feature_channels, local_channels, bn=True)

        # Feature Propagation Layers
        # First build the in_channel for each FP
        if use_xyz:
            sa_channels_copy = [(in_channels, )] + list(sa_channels)
        else:
            sa_channels_copy = [(in_channels - 3, )] + list(sa_channels)
        fchannel1 = local_channels
        feature_channels = []
        while len(sa_channels_copy) != 0:
            fchannel2 = fchannel1
            fchannel1 = sa_channels_copy.pop()
            feature_channels.append(fchannel2[-1] + fchannel1[-1])

        self.fp_modules = nn.ModuleList()
        for ind in range(num_fp_layers):
            fp_module = PointnetFPModule(in_channels=feature_channels[ind],
                                         mlp_channels=fp_channels[ind],
                                         num_neighbors=num_fp_neighbour[ind])
            self.fp_modules.append(fp_module)

        # Fully Connected Layers
        feature_channels = fp_channels[-1][-1]
        self.conv_seg = Conv1d(feature_channels, seg_channels[0], 1)
        self.dropout = nn.Dropout(p=dropout)
        self.seg_logit = nn.Conv1d(seg_channels[0], num_seg_classes, 1, bias=True)

        self.init_weights()
        set_bn(self, momentum=0.01)

    def forward(self, data_batch):
        # TODO: Modify the comment related to the tensor dimension

        point = data_batch["points"]
        end_points = {}

        xyz = point.narrow(1, 0, 3)
        if point.size(1) > 3:
            feature = point.narrow(1, 3, point.size(1) - 3)
        else:
            feature = None
        inter_xyz = [xyz]  # Treat it as stack
        inter_feature = [feature]  # Treat it as stack

        # Set Abstraction Layers
        for sa_module in self.sa_modules:
            xyz, feature = sa_module(inter_xyz[-1], inter_feature[-1])
            inter_xyz.append(xyz)
            inter_feature.append(feature)

        # Local Feature Extraction Layers
        if self.use_xyz:
            feature = torch.cat([xyz, feature], dim=1)
        feature = self.mlp_local(feature)

        # Feature Propagation Layers
        xyz2 = xyz
        features2 = feature
        for fp_module in self.fp_modules:
            xyz1 = inter_xyz.pop()
            features1 = inter_feature.pop()
            if self.use_xyz and len(inter_feature) == 0:
                features1 = point

            new_features1 = fp_module(xyz1, xyz2, features1, features2)

            xyz2 = xyz1
            features2 = new_features1

        # Fully Connected Layers
        x = self.conv_seg(new_features1)
        end_points["feats"] = x

        x = self.dropout(x)
        seg_logit = self.seg_logit(x)

        preds = {
            "seg_logit": seg_logit
        }
        preds.update(end_points)

        return preds

    def init_weights(self):
        nn.init.xavier_uniform_(self.seg_logit.weight)
        nn.init.zeros_(self.seg_logit.bias)

class PointNet2SSGPartSegLoss(nn.Module):
    """Pointnet2 part segmentation loss [Incomplete comment]"""
    def __init__(self, seg_loss_weight):
        super(PointNet2SSGPartSegLoss, self).__init__()
        self.seg_loss_weight = seg_loss_weight
        assert self.seg_loss_weight >= 0.0

    def forward(self, preds, labels):
        seg_logit = preds["seg_logit"]
        seg_label = labels["seg_label"]
        seg_loss = F.cross_entropy(seg_logit, seg_label)
        loss_dict = {
            "seg_loss": seg_loss * self.seg_loss_weight
        }

        return loss_dict

if __name__ == '__main__':
    batch_size = 8
    in_channels = 3
    num_points = 1024
    num_seg_classes = 50

    points = torch.rand(batch_size, in_channels, num_points)
    points = points.cuda()

    pn2ps = PointNet2SSGPartSeg(in_channels, num_seg_classes)
    pn2ps.cuda()
    out_dict = pn2ps({"points": points})
    for k, v in out_dict.items():
        print('PointNet2:', k, v.shape)