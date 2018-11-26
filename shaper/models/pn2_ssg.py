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

from shaper.models.pn2_modules.pointnet2_modules import PointnetSAModule
from shaper.nn import MLP
from shaper.nn.init import set_bn
from shaper.models.metric import Accuracy


class PointNet2SSG_Cls(nn.Module):
    """PointNet2 with single-scale grouping for classfication

    Structure: input -> [PointNetSA] -> [MLP] -> [Linear] -> logits

    Args:
        in_channels: int = 3
            Number of input channels
        out_channels: int
            Number of semantics classes to predict over -- size of softmax classifier
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, in_channels, out_channels,
                 num_points_list=(512, 128), radius_list=(0.2, 0.4),
                 nsamples_list=(32, 64), group_mlps=((64, 64, 128), (128, 128, 256)),
                 global_mlps=(256, 512, 1024), fc_channels=(512, 256),
                 drop_prob=0.5, use_xyz=True):
        super().__init__()

        ssg_layer_num = len(num_points_list)
        assert len(radius_list) == ssg_layer_num
        assert len(nsamples_list) == ssg_layer_num
        assert len(group_mlps) == ssg_layer_num

        feature_channels = in_channels - 3
        self.SA_modules = nn.ModuleList()
        for i, (num_points, radius, nsamples, group_mlps) in enumerate(
                zip(num_points_list, radius_list, nsamples_list, group_mlps)):
            group_mlps = list(group_mlps)
            group_mlps.insert(0, feature_channels)
            self.SA_modules.append(
                PointnetSAModule(npoint=num_points,
                                 radius=radius,
                                 nsample=nsamples,
                                 mlp=group_mlps, use_xyz=use_xyz))
            feature_channels = group_mlps[-1]

        global_mlps = list(global_mlps)
        global_mlps.insert(0, feature_channels)
        self.SA_modules.append(
            PointnetSAModule(mlp=global_mlps, use_xyz=use_xyz))

        fc_in_channels = global_mlps[-1]
        self.mlp_global = MLP(fc_in_channels, fc_channels, dropout=drop_prob)
        self.classifier = nn.Linear(fc_channels[-1], out_channels, bias=True)

        self.init_weights()
        set_bn(self, momentum=0.01)


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, data_batch):
        pointcloud = data_batch["points"]
        pointcloud = pointcloud.transpose(1, 2)
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)
            # if xyz is not None:
            #     print('xyz: ', list(xyz.size()))
            # print('features: ', list(features.size()))
        x = self.mlp_global(features.squeeze(-1))
        cls_logits = self.classifier(x)
        preds = {
            'cls_logits': cls_logits
        }

        return preds

    def init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)


class PointNet2SSG_ClsLoss(nn.Module):
    def __init__(self):
        super(PointNet2SSG_ClsLoss, self).__init__()

    def forward(self, preds, labels):
        cls_logits = preds["cls_logits"]
        cls_labels = labels["cls_labels"]
        cls_loss = F.cross_entropy(cls_logits, cls_labels)
        loss_dict = {
            'cls_loss': cls_loss,
        }
        return loss_dict


def build_pointnet2ssg(cfg):
    if cfg.TASK == "classification":
        net = PointNet2SSG_Cls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            num_points_list=cfg.MODEL.PN2SSG.NUM_POINTS,
            radius_list=cfg.MODEL.PN2SSG.RADIUS,
            nsamples_list=cfg.MODEL.PN2SSG.NUM_SAMPLE,
            group_mlps=cfg.MODEL.PN2SSG.GROUP_MLPS,
            global_mlps=cfg.MODEL.PN2SSG.GLOBAL_MLPS,
            fc_channels=cfg.MODEL.PN2SSG.FC_CHANNELS,
            drop_prob=cfg.MODEL.PN2SSG.DROP_PROB,
            use_xyz=cfg.MODEL.PN2SSG.USE_XYZ
        )
        loss_fn = PointNet2SSG_ClsLoss()
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

    pn2ssg = PointNet2SSG_Cls(in_channels, num_classes, num_points_list=(128, 64),
                              nsamples_list=(128, 128))
    pn2ssg.cuda()
    out_dict = pn2ssg({"points": data})
    for k, v in out_dict.items():
        print('pointnet:', k, v.shape)
