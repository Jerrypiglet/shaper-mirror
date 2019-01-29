"""PointNet for semantic segmentation

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

from shaper.nn import MLP, SharedMLP, Conv1d
from shaper.nn.init import xavier_uniform, set_bn
from shaper.models.pointnet.pointnet_cls import TNet, Stem



class PointNetSemSeg(nn.Module):
    """PointNet for semantic segmentation, based on:
    https://github.com/charlesq34/pointnet/blob/master/sem_seg/model.py
    """

    def __init__(self,
                 in_channels,
                 num_seg_classes,
                 stem_channels=(64, 64),
                 local_channels=(64, 128, 1024),
                 global_channels=(256, 128),
                 seg_channels=(512, 256, 128, 128),
                 dropout_prob=0.3,
                 with_transform=True):
        """
        Args:
           in_channels (int): the number of input channels
           num_seg_classes (int): the number of output channels
           stem_channels (tuple of int): the numbers of channels in stem feature extractor
           local_channels (tuple of int): the numbers of channels in local mlp
           global_channels (tuple of int): the numbers of channels in global mlp
           mlp_seg_channels (tuple of int): the numbers of channels in segmentation mlp
           dropout_prob (float): the probability to dropout
           with_transform (bool): whether to use TNet to transform features.

        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = num_seg_classes

        self.stem = Stem(in_channels, stem_channels, with_transform=with_transform)
        self.mlp_local = SharedMLP(stem_channels[-1], local_channels)

        self.mlp_global = MLP(local_channels[-1], global_channels)

        self.mlp_seg = SharedMLP(local_channels[-1] + global_channels[-1], seg_channels,
                                 dropout=dropout_prob)
        self.mlp_seg_logits = Conv1d(seg_channels[-1], out_channels, 1, relu=False, bn=False)
        
        self.init_weights()

    def init_weights(self):
        self.mlp_local.init_weights(xavier_uniform)
        self.mlp_seg.init_weights(xavier_uniform)
        # Set batch normalization to 0.01 as default
        set_bn(self, momentum=0.01)

    def forward(self, data_batch):
        x = data_batch["points"]

        # stem
        x, end_points = self.stem(x)

        # mlp local
        x_local = self.mlp_local(x)

        # max pooling and mlp global
        x, max_indices = torch.max(x_local, 2)
        end_points['key_point_inds'] = max_indices
        x_global = self.mlp_global(x)

        # concat local and global features
        x = torch.cat([x_local, x_global.unsqueeze(2).expand(-1, -1, x_local.shape[2])], 1)

        # mlp for point features
        x = self.mlp_seg(x)
        x = self.mlp_seg_logits(x)

        preds = {
            'seg_logit': x
        }
        preds.update(end_points)
        return preds
