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
from shaper.models.pointnet2.modules import PointNetSAModuleMSG, PointnetFPModule
from shaper.nn.init import set_bn, xavier_uniform


class PointNet2MSGSemSeg(nn.Module):
    """ PointNet2 with multi-scale grouping for semantic segmentation

    Structure: input -> [PointNetSA(MSG)]s -> [PointNetFP]s -> [FC layer]s
    
    """

    def __init__(self, 
                 in_channels,
                 num_seg_classes,
                 num_centroids=(1024, 256, 64, 16),
                 radius_list=((0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.8)),
                 num_neighbours_list=((16, 32), (16, 32), (16, 32), (16, 32)),
                 sa_channels_list=(
                         ((16, 16, 32), (32, 32, 64)),
                         ((64, 64, 128), (64, 96, 128)),
                         ((128, 196, 256), (128, 196, 256)),
                         ((256, 256, 512), (256, 384, 512))),
                 #local_channels=(256, 512, 1024),
                 fp_channels=((128, 128), (256, 256), (512, 512), (512, 512)),
                 num_fp_neighbours=(3, 3, 3, 3),
                 seg_channels=(128,),
                 dropout_prob=0.5,
                 use_xyz=True):
        """
        Args:
            in_channels (int): the number of input channels
            num_seg_classes (int): the number of segmentation classes
            num_centroids (tuple of int): the numbers of centroids to sample in each set abstraction module
            radius_list (tuple of tuple of float): the radii to query neighbours in each set abstraction module
            num_neighbours (tuple of tuple of int): the numbers of neighbours to query for each centroid
            sa_channels (tuple of tuple of tuple of int): the numbers of mlp channels within each set abstraction module
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

        # sanity check
        num_sa_layers = len(num_centroids)
        num_fp_layers = len(fp_channels)
        assert len(radius_list) == num_sa_layers, \
                """The number of lists of radii should be equal to the size of num_centroids.
                Got {} and {}""".format(len(radius_list), num_sa_layers)
        assert len(num_neighbours_list) == num_sa_layers, \
                """The number of lists of num_neighbors should be equal to the size of num_centroids.
                Got {} and {}""".format(len(num_neighbours_list), num_sa_layers)
        assert len(sa_channels_list) == num_sa_layers, \
                """The number of lists of set abstraction channels should be equal to the size of num_centroids.
                Got {} and {}""".format(len(sa_channels_list), num_sa_layers)
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
            sa_module = PointNetSAModuleMSG(in_channels=feature_channels,
                                            mlp_channels_list=sa_channels_list[ind],
                                            num_centroids=num_centroids[ind],
                                            radius_list=radius_list[ind],
                                            num_neighbours_list=num_neighbours_list[ind],
                                            use_xyz=use_xyz)
            self.sa_modules.append(sa_module)
            feature_channels = sa_module.out_channels

        # Local Feature Extraction Layers (TODO - check original architecture)
        if use_xyz:
            feature_channels += 3
        #self.mlp_local = SharedMLP(feature_channels, local_channels, bn=True)

        # Feature Propagation Layers
        #feature_channels = local_channels[-1]
        inter_channels = [in_channels if use_xyz else in_channels - 3]
        inter_channels.extend([sa_modules.out_channels for sa_modules in self.sa_modules])
        self.fp_modules = nn.ModuleList()
        for ind in range(num_fp_layers):
            fp_module = PointnetFPModule(in_channels=feature_channels + inter_channels[-2 - ind],
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
        # self.mlp_local.init_weights(xavier_uniform)
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
        # feature = self.mlp_local(feature)

        # Feature Propagation Layers
        dense_xyz = xyz
        dense_feature = feature
        for fp_ind, fp_module in enumerate(self.fp_modules):
            sparse_xyz = inter_xyz[-2 - fp_ind]
            sparse_feature = inter_feature[-2 - fp_ind]
            fp_feature = fp_module(sparse_xyz, dense_xyz, sparse_feature, dense_feature)
            dense_xyz = sparse_xyz
            dense_feature = fp_feature

        # Fully Connected Layers
        x = self.mlp_seg(dense_feature)
        seg_logit = self.seg_logit(x)
        # seg_logit.transpose_(0, 1)

        preds = {"seg_logit": seg_logit}
        # preds.update(end_points)
        return preds


if __name__ == '__main__':
    batch_size = 2
    in_channels = 3
    num_points = 1024
    num_seg_classes = 50

    points = torch.rand(batch_size, in_channels, num_points)
    if torch.cuda.is_available():
        points = points.cuda()

    pn2msg = PointNet2MSGSemSeg(in_channels, num_seg_classes)
    if torch.cuda.is_available():
        pn2msg.cuda()
    out_dict = pn2msg({"points": points})
    for k, v in out_dict.items():
        print('PointNet2MSG:', k, v.shape)
