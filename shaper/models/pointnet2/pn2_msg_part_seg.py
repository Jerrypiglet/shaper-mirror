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

from shaper.nn import SharedMLP
from shaper.models.pointnet2.modules import PointNetSAModuleMSG, PointnetFPModule
from shaper.nn.init import set_bn


class PointNet2MSGPartSeg(nn.Module):
    """ PointNet 2 part segmentation with multi-scale grouping

    Structure: input -> [PointNetSA(MSG)]s -> [MLP]s -> [PointNetFP]s -> [FC layer]s

    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 num_seg_classes,
                 num_centroids=(512, 128),
                 radius_list=((0.1, 0.2, 0.4), (0.4, 0.8)),
                 num_neighbours_list=((32, 64, 128), (64, 128)),
                 sa_channels_list=(
                         ((32, 32, 64), (64, 64, 128), (64, 96, 128)),
                         ((128, 128, 256), (128, 196, 256))),
                 local_channels=(256, 512, 1024),
                 fp_channels=((256, 256), (256, 128), (128, 128)),
                 num_fp_neighbours=(3, 3, 3),
                 seg_channels=(128,),
                 dropout_prob=0.5,
                 use_xyz=True):
        """

        Args:
            in_channels:
            num_classes:
            num_seg_classes:
            num_centroids:
            radius_list:
            num_neighbours_list:
            sa_channels_list:
            local_channels:
            fp_channels:
            num_fp_neighbours:
            seg_channels:
            dropout_prob:
            use_xyz:
        """
        super(PointNet2MSGPartSeg, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        self.use_xyz = use_xyz

        # sanity check
        num_sa_layers = len(num_centroids)
        num_fp_layers = len(fp_channels)
        assert len(radius_list) == num_sa_layers
        assert len(num_neighbours_list) == num_sa_layers
        assert len(sa_channels_list) == num_sa_layers
        assert (num_sa_layers + 1) == num_fp_layers  # SA layer + Local feature extraction layer = FP layers
        assert len(num_fp_neighbours) == num_fp_layers

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

        # Local Feature Extraction Layers
        if use_xyz:
            feature_channels += 3
        self.mlp_local = SharedMLP(feature_channels, local_channels, bn=True)

        # Feature Propagation Layers
        feature_channels = local_channels[-1]
        inter_channels = [in_channels if use_xyz else in_channels - 3]
        inter_channels[0] += num_classes  # concat with one-hot
        inter_channels.extend([sa_modules.out_channels for sa_modules in self.sa_modules])
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
        self.seg_logit = nn.Conv1d(seg_channels[-1], num_seg_classes, 1, bias=True)

        self.init_weights()
        set_bn(self, momentum=0.01)

    def forward(self, data_batch):
        points = data_batch["points"]
        end_points = {}

        xyz = points.narrow(1, 0, 3)
        if points.size(1) > 3:
            feature = points.narrow(1, 3, points.size(1) - 3)
        else:
            feature = None

        # Save intermediate results
        inter_xyz = [xyz]
        inter_feature = [points if self.use_xyz else feature]

        # Create one hot class label
        num_points = points.shape[2]
        with torch.no_grad():
            cls_label = data_batch["cls_label"]
            I = torch.eye(self.num_classes, dtype=points.dtype, device=points.device)
            one_hot = I[cls_label]
            one_hot_expand = one_hot.unsqueeze(2).expand(-1, -1, num_points)
            inter_feature[0] = torch.cat([inter_feature[0], one_hot_expand], dim=1)

        # Set Abstraction Layers
        for sa_module in self.sa_modules:
            xyz, feature = sa_module(xyz, feature)
            inter_xyz.append(xyz)
            inter_feature.append(feature)

        # Local Feature Extraction Layers
        if self.use_xyz:
            feature = torch.cat([xyz, feature], dim=1)
        feature = self.mlp_local(feature)

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

        preds = {
            "seg_logit": seg_logit
        }
        preds.update(end_points)

        return preds

    def init_weights(self):
        nn.init.xavier_uniform_(self.seg_logit.weight)
        nn.init.zeros_(self.seg_logit.bias)


if __name__ == '__main__':
    batch_size = 2
    in_channels = 3
    num_points = 1024
    num_classes = 16
    num_seg_classes = 50

    points = torch.rand(batch_size, in_channels, num_points)
    points = points.cuda()
    cls_label = torch.randint(num_classes, (batch_size,))
    cls_label = cls_label.cuda()

    pn2msg = PointNet2MSGPartSeg(in_channels, num_classes, num_seg_classes)
    pn2msg.cuda()
    out_dict = pn2msg({"points": points, "cls_label": cls_label})
    for k, v in out_dict.items():
        print('PointNet2MSG:', k, v.shape)
