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

from shaper.nn import SharedMLP, Conv1d
from shaper.models.pointnet2.modules import PointNetSAModule, PointnetFPModule
from shaper.nn.init import set_bn, xavier_uniform


class PointNet2SSGSemSeg(nn.Module):
    """PointNet 2 semantic segmentation with single-scale grouping.

    Structure: input -> [PointNetSA]s -> [PointNetFP]s -> [FC layer]s

    PointNetSA: PointNet Set Abstraction Layer
    PointNetFP: PointNet Feature Propagation Layer
    """

    def __init__(self,
                 in_channels,
                 num_seg_classes,
                 num_centroids=(1024, 256, 64, 16),
                 radius=(0.1, 0.2, 0.4, 0.8),
                 num_neighbours=(32, 32, 32, 32),
                 sa_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256, 512)),
                 fp_channels=((256, 256), (256, 256), (256, 128), (128, 128, 128)),
                 num_fp_neighbours=(3, 3, 3),
                 seg_channels=(128,),
                 dropout_prob=0.5,
                 use_xyz=True):
        """
        Args:
            in_channels (int): the number of input channels
            num_seg_classes (int): the number of segmentation classes
            num_centroids (tuple of int): the numbers of centroids to sample in each set abstraction module
            radius (tuple of float): the radii to query neighbours in each set abstraction module
            num_neighbours (tuple of int): the numbers of neighbours to query for each centroid
            sa_channels (tuple of tuple of int): the numbers of mlp channels within each set abstraction module
            fp_channels (tuple of tuple of int): the numbers of mlp channels for feature propagation (FP) module
            num_fp_neighbours (tuple of int): the numbers of nearest neighbors used in FP
            seg_channels (tuple of int): the numbers of channels in segmentation mlp
            dropout_prob (float): the probability to dropout input features
            use_xyz (bool): whether or not to use the xyz position of a points as a feature
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_seg_classes = num_seg_classes
        self.use_xyz = use_xyz

        # Sanity check
        num_sa_layers = len(num_centroids)
        num_fp_layers = len(fp_channels)
        assert len(radius) == num_sa_layers, \
                """The number of lists of radii should be equal to the size of num_centroids.
                Got {} and {}""".format(len(radius), num_sa_layers)
        assert len(num_neighbours) == num_sa_layers, \
                """The number of lists of num_neighbors should be equal to the size of num_centroids.
                Got {} and {}""".format(len(num_neighbours), num_sa_layers)
        assert len(sa_channels) == num_sa_layers, \
                """The number of lists of set abstraction channels should be equal to the size of num_centroids.
                Got {} and {}""".format(len(sa_channels), num_sa_layers)
        assert num_sa_layers == num_fp_layers, \
                """The size of num_centroids should be equal to the size of the list of fp channels.
                Got {} and {}""".format(num_sa_layers, num_fp_layers)
        assert len(num_fp_neighbours) == num_fp_layers, \
                """The size of num_fp_neighbours should be equal to the size of the list of fp channels.
                Got {} and {}""".format(num_sa_layers, num_fp_layers)
        

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

        if use_xyz:
            feature_channels += 3

        # Feature Propagation Layers
        inter_channels = [in_channels if use_xyz else in_channels - 3]
        inter_channels[0] += num_classes  # concat with one-hot
        inter_channels.extend([x[-1] for x in sa_channels])
        self.fp_modules = nn.ModuleList()
        for ind in range(num_fp_layers):
            fp_module = PointnetFPModule(in_channels=feature_channels + inter_channels[-1 - ind],
                                         mlp_channels=fp_channels[ind],
                                         num_neighbors=num_fp_neighbours[ind])
            self.fp_modules.append(fp_module)
            feature_channels = fp_channels[ind][-1]

        # Fully Connected Layers
        feature_channels = fp_channels[-1][-1]
        self.mlp_seg = SharedMLP(feature_channels, seg_channels, ndim=1, dropout=dropout_prob)
        self.seg_logit = Conv1d(seg_channels[-1], num_seg_classes, 1, relu=False, bn=False)

        self.init_weights()

    def init_weights(self):
        for sa_module in self.sa_modules:
            sa_module.init_weights(xavier_uniform)
        for fp_module in self.fp_modules:
            fp_module.init_weights(xavier_uniform)
        self.mlp_seg.init_weights(xavier_uniform)
        self.mlp_seg.init_weights(xavier_uniform)
        set_bn(self, momentum=0.01)

    def forward(self, data_batch):
        points = data_batch["points"]
        # end_points = {}

        # Break up pointcloud
        xyz = points.narrow(1, 0, 3)
        if points.size(1) > 3:
            feature = points.narrow(1, 3, points.size(1) - 3)
        else:
            feature = None

        # Save intermediate results
        inter_xyz = [xyz]
        inter_feature = [points if self.use_xyz else feature]

        # Set Abstraction Layers
        for sa_module in self.sa_modules:
            xyz, feature = sa_module(xyz, feature)
            inter_xyz.append(xyz)
            inter_feature.append(feature)

        # Local Feature Extraction Layers
        if self.use_xyz:
            feature = torch.cat([xyz, feature], dim=1)

        # Feature Propagation Layers
        dense_xyz = xyz
        dense_feature = feature
        for fp_ind, fp_module in enumerate(self.fp_modules):
            sparse_xyz = inter_xyz[-1 - fp_ind]
            sparse_feature = inter_feature[-1 - fp_ind]
            fp_feature = fp_module(sparse_xyz, dense_xyz, sparse_feature, dense_feature)
            dense_xyz = sparse_xyz
            dense_feature = fp_feature

        # Fully Connected Layers
        x = self.mlp_seg(dense_feature)
        seg_logit = self.seg_logit(x)

        preds = {"seg_logit": seg_logit}
        # preds.update(end_points)
        return preds


if __name__ == '__main__':
    batch_size = 8
    in_channels = 3
    num_points = 1024
    num_seg_classes = 50

    points = torch.rand(batch_size, in_channels, num_points)
    points = points.cuda()

    pn2ssg = PointNet2SSGSemSeg(in_channels, num_seg_classes)
    pn2ssg.cuda()
    out_dict = pn2ssg({"points": points})
    for k, v in out_dict.items():
        print('PointNet2SSG:', k, v.shape)
