"""
PointNet++

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

from shaper.models.pn2_modules.pointnet2_modules import PointnetSAModuleMSG, PointnetSAModule
from shaper.models._utils import FC


class Pointnet2MSG_Cls(nn.Module):
    """
        PointNet2 with multi-scale grouping
        Classification network

        Parameters
        ----------
        input_channels: int = 3
            Number of input channels
        out_channels: int
            Number of semantics classes to predict over -- size of softmax classifier
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels, out_channels,
                 num_points_list=(512, 128), radius_list=((0.1, 0.2, 0.4), (0.2, 0.4, 0.8)),
                 nsamples_list=((16, 32, 128), (32, 64, 128)),
                 group_mlps_list=(((32, 32, 64), (64, 64, 128), (64, 96, 128)), ((64, 64, 128), (128, 128, 256), (128, 128, 256))),
                 global_mlps=(256, 512, 1024), fc_channels=(512, 256    ), drop_prob=0.5,
                 use_xyz=True, bn=True):
        super().__init__()

        msg_layer_num = len(num_points_list)
        assert len(radius_list) == msg_layer_num
        assert len(nsamples_list) == msg_layer_num
        assert len(group_mlps_list) == msg_layer_num

        feature_channels = input_channels-3
        self.SA_modules = nn.ModuleList()
        for i, (num_points, radius, nsamples, mlps) in enumerate(zip(num_points_list, radius_list, nsamples_list, group_mlps_list)):
            sma_layer_num = len(radius)
            assert len(nsamples) == sma_layer_num
            assert len(mlps) == sma_layer_num
            # sam_mlps = [list(_).insert(0, feature_channels) for _ in mlps]
            group_sam_mlps = []
            for _ in mlps:
                _ = list(_)
                _.insert(0, feature_channels)
                group_sam_mlps.append(_)
            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=num_points,
                    radii=list(radius),
                    nsamples=list(nsamples),
                    mlps=group_sam_mlps,
                    use_xyz=use_xyz
                )
            )
            feature_channels = 0
            for _ in mlps:
                feature_channels += _[-1]
        global_mlps = list(global_mlps)
        global_mlps.insert(0, feature_channels)
        self.SA_modules.append(
            PointnetSAModule(
                mlp=global_mlps, use_xyz=use_xyz
            )
        )
        fc_in_channels = global_mlps[-1]
        FC_layers = []
        for fc_out_channels in fc_channels:
            FC_layers.append(FC(fc_in_channels, fc_out_channels, bn=bn))
            FC_layers.append(nn.Dropout(p=drop_prob))
            fc_in_channels = fc_out_channels
        FC_layers.append(nn.Linear(fc_in_channels, out_channels))
        self.FC_layer = nn.Sequential(*FC_layers)

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
        cls_logits = self.FC_layer(features.squeeze(-1))
        preds = {
            'cls_logits': cls_logits
        }

        return preds

class PointNet2MSG_ClsLoss(nn.Module):
    def __init__(self):
        super(PointNet2MSG_ClsLoss, self).__init__()

    def forward(self, preds, labels):
        cls_logits = preds["cls_logits"]
        cls_labels = labels["cls_labels"]
        cls_loss = F.cross_entropy(cls_logits, cls_labels)
        loss_dict = {
            'cls_loss': cls_loss,
        }
        return loss_dict

class PointNet2MSG_Metric(nn.Module):
    def forward(self, preds, labels):
        cls_logits = preds["cls_logits"]
        cls_labels = labels["cls_labels"]
        pred_labels = cls_logits.argmax(1)
        acc = pred_labels.eq(cls_labels).float().mean()

        metric_dict = {
            'acc': acc
        }
        return metric_dict

def build_pointnet2msg(cfg):
    if cfg.TASK == "classification":
        net = Pointnet2MSG_Cls(
            input_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            num_points_list=cfg.MODEL.PN2MSG.NUM_POINTS,
            radius_list=cfg.MODEL.PN2MSG.RADIUS,
            nsamples_list=cfg.MODEL.PN2MSG.NUM_SAMPLE,
            group_mlps_list=cfg.MODEL.PN2MSG.GROUP_MLPS,
            global_mlps=cfg.MODEL.PN2MSG.GLOBAL_MLPS,
            fc_channels=cfg.MODEL.PN2MSG.FC_CHANNELS,
            drop_prob=cfg.MODEL.PN2MSG.DROP_PROB,
            use_xyz=cfg.MODEL.PN2MSG.USE_XYZ
        )
        loss_fn = PointNet2MSG_ClsLoss()
        metric_fn = PointNet2MSG_Metric()
    else:
        raise NotImplementedError

    return net, loss_fn, metric_fn

if __name__ == '__main__':
    batch_size = 8
    in_channels = 3
    num_points = 1024
    num_classes = 40

    data = torch.rand(batch_size, num_points, in_channels)
    data = data.cuda()
    # data.cuda()
    pn2msg = Pointnet2MSG_Cls(3, 40)
    pn2msg.cuda()
    out_dict = pn2msg({"points": data})
    for k, v in out_dict.items():
        print('pointnet:', k, v.shape)
