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
from shaper.models.pn2_utils import PointNetSAModuleMSG
from shaper.nn.init import set_bn
from shaper.models.loss import ClsLoss
from shaper.models.metric import Accuracy


class PointNet2MSGCls(nn.Module):
    """PointNet2 with multi-scale grouping for classification

    Structure: input -> [PointNetSA(MSG)]s -> [MLP]s -> MaxPooling -> [MLP]s -> [Linear] -> logits

    Args:
        Refer to PointNet2SSGCls. Major difference is that all the arguments will be a tuple of original types.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_centroids=(512, 128),
                 radius_list=((0.1, 0.2, 0.4), (0.2, 0.4, 0.8)),
                 num_neighbours_list=((16, 32, 128), (32, 64, 128)),
                 sa_channels_list=(
                         ((32, 32, 64), (64, 64, 128), (64, 96, 128)),
                         ((64, 64, 128), (128, 128, 256), (128, 128, 256))),
                 local_channels=(256, 512, 1024),
                 global_channels=(512, 256),
                 dropout_prob=0.5,
                 use_xyz=True):
        super(PointNet2MSGCls, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_xyz = use_xyz

        # sanity check
        num_layers = len(num_centroids)
        assert len(radius_list) == num_layers
        assert len(num_neighbours_list) == num_layers
        assert len(sa_channels_list) == num_layers

        feature_channels = in_channels - 3
        self.sa_modules = nn.ModuleList()
        for ind in range(num_layers):
            sa_module = PointNetSAModuleMSG(in_channels=feature_channels,
                                            mlp_channels_list=sa_channels_list[ind],
                                            num_centroids=num_centroids[ind],
                                            radius_list=radius_list[ind],
                                            num_neighbours_list=num_neighbours_list[ind],
                                            use_xyz=use_xyz)
            self.sa_modules.append(sa_module)
            feature_channels = sa_module.out_channels

        if use_xyz:
            feature_channels += 3
        self.mlp_local = SharedMLP(feature_channels, local_channels, bn=True)
        self.mlp_global = MLP(local_channels[-1], global_channels, dropout=dropout_prob)
        self.classifier = nn.Linear(global_channels[-1], out_channels, bias=True)

        self.init_weights()
        set_bn(self, momentum=0.01)

    def forward(self, data_batch):
        point = data_batch["points"]
        end_points = {}

        # torch.Tensor.narrow; share same memory
        xyz = point.narrow(1, 0, 3)
        if point.size(1) > 3:
            feature = point.narrow(1, 3, point.size(1) - 3)
        else:
            feature = None

        for sa_module in self.sa_modules:
            xyz, feature = sa_module(xyz, feature)

        if self.use_xyz:
            x = torch.cat([xyz, feature], dim=1)
        else:
            x = feature
        x = self.mlp_local(x)
        x, max_indices = torch.max(x, 2)
        end_points['key_point_inds'] = max_indices
        x = self.mlp_global(x)

        cls_logits = self.classifier(x)

        preds = {
            'cls_logits': cls_logits
        }
        preds.update(end_points)

        return preds

    def init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)


def build_pointnet2msg(cfg):
    if cfg.TASK == "classification":
        net = PointNet2MSGCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            num_centroids=cfg.MODEL.PN2MSG.NUM_CENTROIDS,
            radius_list=cfg.MODEL.PN2MSG.RADIUS,
            num_neighbours_list=cfg.MODEL.PN2MSG.NUM_NEIGHBOURS,
            sa_channels_list=cfg.MODEL.PN2MSG.SA_CHANNELS,
            global_channels=cfg.MODEL.PN2MSG.GLOBAL_CHANNELS,
            dropout_prob=cfg.MODEL.PN2MSG.DROPOUT_PROB,
            use_xyz=cfg.MODEL.PN2MSG.USE_XYZ
        )
        loss_fn = ClsLoss()
        metric_fn = Accuracy()
    else:
        raise NotImplementedError

    return net, loss_fn, metric_fn


if __name__ == '__main__':
    batch_size = 8
    in_channels = 3
    num_points = 1024
    num_classes = 40

    data = torch.rand(batch_size, in_channels, num_points)
    data = data.cuda()

    pn2msg = PointNet2MSGCls(in_channels, num_classes)
    pn2msg.cuda()
    out_dict = pn2msg({"points": data})
    for k, v in out_dict.items():
        print('PointNet2MSG:', k, v.shape)
