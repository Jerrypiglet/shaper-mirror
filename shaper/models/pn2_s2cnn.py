"""
PointNet++ + Local Spherical CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.nn.functional import smooth_cross_entropy
from shaper.models.pn2_modules import pointnet2_utils
from shaper.models.pn2_modules.pointnet2_modules import PointnetSAModule
from shaper.models.s2cnn import S2CNNFeature
from shaper.models.metric import Accuracy
from shaper.models.dg_pn2 import TNet
from shaper.nn import MLP, SharedMLP
from shaper.nn.init import set_bn


class PNS2CNNCls(nn.Module):
    """Dynamic Graph + Local Spherical CNN for classification

    Structure: input -> [S2CNN] (-> [TNet] -> transform_group_center)
    -> [Concat] -> [PointNet] -> logits

    Attributes:
        transform_xyz： whether to transform group center coordinates

    """

    def __init__(self, in_channels, out_channels,
                 num_points=16, radius_list=(0.2,), num_samples_list=(16,),
                 band_width_in_list=(16,), s2cnn_feature_channels_list=((32, 64),),
                 band_width_list=((16, 8),), k=4,
                 global_mlps=(256, 512, 1024), fc_channels=(512, 256), drop_prob=0.5,
                 use_xyz=True, transform_xyz=True):

        super(PNS2CNNCls, self).__init__()
        assert (in_channels in [3, 6])
        self.local_group_scale_num = len(radius_list)
        assert len(num_samples_list) == self.local_group_scale_num
        assert len(band_width_in_list) == self.local_group_scale_num
        assert len(s2cnn_feature_channels_list) == self.local_group_scale_num
        assert len(band_width_list) == self.local_group_scale_num

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_normal = False
        if in_channels == 6:
            self.use_normal = True

        # local grouping
        self.npoint = num_points
        self.num_samples_list = num_samples_list
        self.groupers = nn.ModuleList()
        for i in range(self.local_group_scale_num):
            radius = radius_list[i]
            nsample = num_samples_list[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroupWithCnt(radius, nsample, use_xyz=True))

        # local s2cnn
        self.band_width_in_list = band_width_in_list
        self.s2cnn_feature_channels_list = s2cnn_feature_channels_list
        self.band_width_list = band_width_list
        self.local_s2cnn_list = nn.ModuleList()
        concat_feature_channels = 3  # xyz
        for i in range(self.local_group_scale_num):
            self.local_s2cnn_list.append(
                S2CNNFeature(in_channels, band_width_in=self.band_width_in_list[i],
                             feature_channels=self.s2cnn_feature_channels_list[i],
                             band_width_list=self.band_width_list[i]))
            concat_feature_channels += self.local_s2cnn_list[i].out_channels

        self.transform_xyz = transform_xyz

        if self.transform_xyz:
            self.transform_input = TNet(concat_feature_channels, 3, k=k)

        global_mlps = list(global_mlps)

        self.mlp_local = SharedMLP(concat_feature_channels, global_mlps)

        fc_in_channels = global_mlps[-1]
        self.mlp_global = MLP(fc_in_channels, fc_channels, dropout=drop_prob)
        self.classifier = nn.Linear(fc_channels[-1], out_channels, bias=True)

        self.init_weights()

    def forward(self, data_batch):
        end_points = {}
        pointcloud = data_batch["points"]
        xyz = pointcloud.narrow(1, 0, 3).contiguous()  # [b, 3, n]
        xyz_flipped = xyz.transpose(1, 2).contiguous()  # [b, n, 3]
        if self.use_normal:
            features = pointcloud.narrow(1, 3, 6).contiguous()
        else:
            features = None
        if pointcloud.size(2) == self.npoint:
            new_xyz = xyz
        else:
            new_xyz = pointnet2_utils.gather_operation(
                xyz,
                pointnet2_utils.furthest_point_sample(xyz_flipped, self.npoint)
            ).contiguous()
        new_xyz_flipped = new_xyz.transpose(1, 2).contiguous()  # [b, np, 3]

        batch_size = pointcloud.size(0)

        local_s2cnn_features_list = []
        for i in range(self.local_group_scale_num):
            new_features, pts_cnt = self.groupers[i](
                xyz_flipped, new_xyz_flipped, features
            )  # new_features: [b, c, np, ns], pts_cnt: [b, np]
            # new_features = new_features.view(batch_size, self.in_channels, -1)
            new_features = new_features.transpose(1, 2).contiguous()
            new_features = new_features.view(-1, self.in_channels, self.num_samples_list[i])

            pts_cnt = pts_cnt.view(-1)
            local_s2cnn_features = self.local_s2cnn_list[i](new_features, pts_cnt)
            local_s2cnn_features = local_s2cnn_features.view(batch_size, self.npoint, -1)
            local_s2cnn_features = local_s2cnn_features.transpose(1, 2).contiguous()

            local_s2cnn_features_list.append(local_s2cnn_features)

        local_s2cnn_features_list_tmp = local_s2cnn_features_list.copy()
        local_s2cnn_features_list_tmp.append(new_xyz)
        x = torch.cat(local_s2cnn_features_list_tmp, dim=1)

        if self.transform_xyz:
            trans_matrix_xyz = self.transform_input(x)
            new_xyz = torch.bmm(trans_matrix_xyz, new_xyz)
            end_points["trans_matrix_xyz"] = trans_matrix_xyz
            local_s2cnn_features_list.append(new_xyz)
            x = torch.cat(local_s2cnn_features_list, dim=1)

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


class PNS2CNNClsLoss(nn.Module):

    def __init__(self, trans_reg_weight):
        super(PNS2CNNClsLoss, self).__init__()
        self.trans_reg_weight = trans_reg_weight

    def forward(self, preds, labels):
        cls_logits = preds["cls_logits"]
        cls_labels = labels["cls_labels"]
        cls_loss = F.cross_entropy(cls_logits, cls_labels)
        loss_dict = {
            'cls_loss': cls_loss,
        }

        # regularization over transform matrix
        if self.trans_reg_weight > 0:
            trans_xyz = preds["trans_matrix_xyz"]
            trans_norm = torch.bmm(trans_xyz, trans_xyz.transpose(2, 1))  # [out, out]
            I = torch.eye(trans_norm.size()[1], dtype=trans_norm.dtype, device=trans_norm.device)
            reg_loss = F.mse_loss(trans_norm, I.unsqueeze(0).repeat(trans_norm.size(0), 1, 1))
            loss_dict["reg_loss"] = reg_loss

        return loss_dict


def build_pns2cnn(cfg):
    if cfg.TASK == "classification":
        net = PNS2CNNCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            num_points=cfg.MODEL.PNS2CNN.NUM_POINTS,
            radius_list=cfg.MODEL.PNS2CNN.RADIUS_LIST,
            num_samples_list=cfg.MODEL.PNS2CNN.NUM_SAMPLE_LIST,
            band_width_in_list=cfg.MODEL.PNS2CNN.BAND_WIDTH_IN_LIST,
            s2cnn_feature_channels_list=cfg.MODEL.PNS2CNN.FEATURE_CHANNELS_LIST,
            band_width_list=cfg.MODEL.PNS2CNN.BAND_WIDTH_LIST,
            k=cfg.MODEL.PNS2CNN.K,
            global_mlps=cfg.MODEL.PNS2CNN.GLOBAL_CHANNELS,
            fc_channels=cfg.MODEL.PNS2CNN.FC_CHANNELS,
            drop_prob=cfg.MODEL.PNS2CNN.DROP_PROB,
            transform_xyz=cfg.MODEL.PNS2CNN.TRANSFORM_XYZ)

        loss_fn = PNS2CNNClsLoss(
            trans_reg_weight=cfg.MODEL.PNS2CNN.TRANS_REG_WEIGHT
        )
        metric_fn = Accuracy()

    else:
        raise NotImplementedError

    return net, loss_fn, metric_fn


if __name__ == "__main__":
    batch_size = 2
    in_channels = 3
    num_points = 1024
    num_classes = 40

    data = torch.rand(batch_size, in_channels, num_points)
    data = data.cuda()

    pns2cnn = PNS2CNNCls(3, 40)
    pns2cnn = pns2cnn.cuda()

    out_dict = pns2cnn({"points": data})
    for k, v in out_dict.items():
        print("pns2cnn: ", k, v.size())
