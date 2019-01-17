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

from shaper.nn import MLP, SharedMLP
from shaper.models.pointnet2.modules import PointNetSAModule
from shaper.nn.init import xavier_uniform, set_bn


class PointNet2SSGCls(nn.Module):
    """PointNet2 with single-scale grouping for classification

    Structure: input -> [PointNetSA]s -> [MLP]s -> [MaxPooling] -> [MLP]s -> [Linear] -> logits
    Notice different with the original implementation, the last set abstraction is implemented as a local MLP.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_centroids=(512, 128),
                 radius=(0.2, 0.4),
                 num_neighbours=(32, 64),
                 sa_channels=((64, 64, 128), (128, 128, 256)),
                 local_channels=(256, 512, 1024),
                 global_channels=(512, 256),
                 dropout_prob=0.5,
                 use_xyz=True):
        """

        Args:
            in_channels (int): the number of input channels
            out_channels (int): the number of semantics classes to predict over
            num_centroids (tuple of int): the numbers of centroids to sample in each set abstraction module
            radius (tuple of float): a tuple of radius to query neighbours in each set abstraction module
            num_neighbours (tuple of int): the numbers of neighbours to query for each centroid
            sa_channels (tuple of tuple of int): the numbers of channels to within each set abstraction module
            local_channels (tuple of int): the numbers of channels to extract local features after set abstraction
            global_channels (tuple of int): the numbers of channels to extract global features
            dropout_prob (float): the probability to dropout input features
            use_xyz (bool): whether or not to use the xyz position of a points as a feature

        """
        super(PointNet2SSGCls, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_xyz = use_xyz

        # sanity check
        num_layers = len(num_centroids)
        assert len(radius) == num_layers
        assert len(num_neighbours) == num_layers
        assert len(sa_channels) == num_layers

        feature_channels = in_channels - 3
        self.sa_modules = nn.ModuleList()
        for ind in range(num_layers):
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
        self.mlp_local = SharedMLP(feature_channels, local_channels, bn=True)
        self.mlp_global = MLP(local_channels[-1], global_channels, dropout=dropout_prob)
        self.classifier = nn.Linear(global_channels[-1], out_channels, bias=True)

        self.init_weights()

    def forward(self, data_batch):
        points = data_batch["points"]
        end_points = {}

        # torch.Tensor.narrow; share same memory
        xyz = points.narrow(1, 0, 3)
        if points.size(1) > 3:
            feature = points.narrow(1, 3, points.size(1) - 3)
        else:
            feature = None

        for sa_module in self.sa_modules:
            xyz, feature = sa_module(xyz, feature)

        if self.use_xyz:
            feature = torch.cat([xyz, feature], dim=1)
        x = self.mlp_local(feature)
        x, max_indices = torch.max(x, 2)
        end_points['key_point_inds'] = max_indices
        x = self.mlp_global(x)

        cls_logit = self.classifier(x)

        preds = {
            'cls_logit': cls_logit
        }
        preds.update(end_points)

        return preds

    def init_weights(self):
        for sa_module in self.sa_modules:
            sa_module.init_weights(xavier_uniform)
        self.mlp_local.init_weights(xavier_uniform)
        self.mlp_global.init_weights(xavier_uniform)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        set_bn(self, momentum=0.01)


if __name__ == '__main__':
    batch_size = 8
    in_channels = 3
    num_points = 1024
    num_classes = 40

    points = torch.rand(batch_size, in_channels, num_points)
    points = points.cuda()

    pn2ssg = PointNet2SSGCls(in_channels, num_classes)
    pn2ssg.cuda()
    out_dict = pn2ssg({"points": points})
    for k, v in out_dict.items():
        print('PointNet2SSG:', k, v.shape)
