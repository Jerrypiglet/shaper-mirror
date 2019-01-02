"""
PointNet++ + Local Spherical CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.models.s2cnn import S2CNNFeature
from shaper.models.metric import Accuracy
from shaper.nn import MLP, SharedMLP
from shaper.models.pn2_utils.modules import QueryGrouperWithCnt, FarthestPointSampler
from shaper.models.pn2_utils import functions as _F
from shaper.models.dgcnn_utils import get_edge_feature


class TNet(nn.Module):
    """Transformation Network for Global DGCNN

    Structure: input -> [EdgeFeature] -> [EdgeConv]s -> [EdgePool] -> features -> [MLP] -> local features
    -> [MaxPool] -> gloal features -> [MLP] -> [Linear] -> Transform matrix

    Args:
        conv_channels (tuple of int): the numbers of channels of edge convolution layers
        k: k-nn for edge feature extractor

    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 conv_channels=(64, 128),
                 local_channels=(1024,),
                 global_channels=(512, 256),
                 k=20):
        super(TNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        self.edge_conv = SharedMLP(2 * in_channels, conv_channels, ndim=2)
        self.mlp_local = SharedMLP(conv_channels[-1], local_channels)
        self.mlp_global = MLP(local_channels[-1], global_channels)
        self.linear = nn.Linear(global_channels[-1], self.out_channels * out_channels, bias=True)

        self.init_weights()

    def forward(self, x):
        """TNet forward

        Args:
            x (torch.Tensor): (batch_size, in_channels, num_points)

        Returns:
            torch.Tensor: (batch_size, out_channels, out_channels)

        """
        x = get_edge_feature(x, self.k)  # (batch_size, 2 * in_channels, num_points, k)
        x = self.edge_conv(x)
        x, _ = torch.max(x, 3)  # (batch_size, edge_channels[-1], num_points)
        x = self.mlp_local(x)
        x, _ = torch.max(x, 2)  # (batch_size, local_channels[-1], num_points)
        x = self.mlp_global(x)
        x = self.linear(x)
        x = x.view(-1, self.out_channels, self.out_channels)
        I = torch.eye(self.out_channels, self.out_channels, device=x.device)
        x.add_(I)  # broadcast first dimension
        return x

    def init_weights(self):
        # set linear transform be 0
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


class PN2S2CNNCls(nn.Module):
    """Dynamic Graph + Local Spherical CNN for classification

    Structure: input -> [S2CNN] (-> [TNet] -> transform_group_center)
    -> [Concat] -> [PointNet] -> logits

    Attributes:
        transform_xyz： whether to transform group center coordinates

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_centroids=24,
                 radius_list=(0.2,),
                 num_neighbours_list=(16,),
                 band_width_in_list=(16,),
                 s2cnn_feature_channels_list=((32, 64),),
                 band_width_list=((16, 8),),
                 k=4,
                 local_channels=(256, 512, 1024),
                 global_channels=(512, 256),
                 dropout_prob=0.5,
                 transform_xyz=True):

        super(PN2S2CNNCls, self).__init__()
        assert (in_channels in [3, 6])
        self.local_group_scale_num = len(radius_list)
        assert len(num_neighbours_list) == self.local_group_scale_num
        assert len(band_width_in_list) == self.local_group_scale_num
        assert len(s2cnn_feature_channels_list) == self.local_group_scale_num
        assert len(band_width_list) == self.local_group_scale_num

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius_list = radius_list

        self.use_normal = False
        if in_channels == 6:
            self.use_normal = True

        # local grouping
        self.num_centroids = num_centroids
        self.sampler = FarthestPointSampler(num_centroids)
        self.num_neighbours_list = num_neighbours_list
        self.groupers = nn.ModuleList()
        for i in range(self.local_group_scale_num):
            radius = radius_list[i]
            num_neighbours = num_neighbours_list[i]
            self.groupers.append(QueryGrouperWithCnt(radius, num_neighbours, use_xyz=True))

        # local s2cnn
        self.band_width_in_list = band_width_in_list
        self.s2cnn_feature_channels_list = s2cnn_feature_channels_list
        self.band_width_list = band_width_list
        self.local_s2cnn_list = nn.ModuleList()
        self.concat_feature_channels = 3  # xyz
        for i in range(self.local_group_scale_num):
            self.local_s2cnn_list.append(
                S2CNNFeature(in_channels, band_width_in=self.band_width_in_list[i],
                             feature_channels=self.s2cnn_feature_channels_list[i],
                             band_width_list=self.band_width_list[i]))
            self.concat_feature_channels += self.local_s2cnn_list[i].out_channels

        self.transform_xyz = transform_xyz

        if self.transform_xyz:
            self.transform_input = TNet(self.concat_feature_channels, 3, k=k)
            self.concat_feature_channels += 3

        local_channels = list(local_channels)

        self.mlp_local = SharedMLP(self.concat_feature_channels, local_channels)

        self.mlp_global = MLP(local_channels[-1], global_channels, dropout=dropout_prob)
        self.classifier = nn.Linear(global_channels[-1], out_channels, bias=True)

        # self.init_weights()

    def forward(self, data_batch):
        end_points = {}
        point = data_batch["points"]
        xyz = point.narrow(1, 0, 3).contiguous()  # [b, 3, n]
        # xyz_flipped = xyz.transpose(1, 2).contiguous()  # [b, n, 3]
        if self.use_normal:
            features = point.narrow(1, 3, 6).contiguous()
        else:
            features = None
        if point.size(2) == self.num_centroids:
            new_xyz = xyz
        else:
            index = self.sampler(xyz)
            new_xyz = _F.gather_points(xyz, index)

        batch_size = point.size(0)

        local_s2cnn_features_list = []
        for i in range(self.local_group_scale_num):
            new_features, pts_cnt = self.groupers[i](
                new_xyz, xyz, features)  # new_features: [b, c, nc, nn], pts_cnt: [b, nc]

            new_features = new_features.transpose(1, 2).contiguous()
            new_features = torch.div(new_features, self.radius_list[i])
            new_features = new_features.view(batch_size * self.num_centroids, self.in_channels,
                                             self.num_neighbours_list[i])

            # import numpy as np
            # for ii in range(10):
            #     np.savetxt("local_pt_{}.txt".format(ii), new_features[ii, ...].detach().cpu().numpy().transpose(), fmt="%.2f")
            # np.savetxt("pts_cnt.txt", pts_cnt.detach().cpu().numpy(), fmt="%d")

            pts_cnt = pts_cnt.view(batch_size * self.num_centroids)
            local_s2cnn_features = self.local_s2cnn_list[i](new_features, pts_cnt)
            # local_s2cnn_features = self.local_s2cnn_list[i](new_features)
            local_s2cnn_features = local_s2cnn_features.view(batch_size, self.num_centroids, -1)

            # for ii in range(10):
            #     np.savetxt("local_feature_{}.txt".format(ii), local_s2cnn_features[0, ii, ...].detach().cpu().numpy(), fmt="%.2f")
            # raise ValueError("step 0")

            local_s2cnn_features = local_s2cnn_features.transpose(1, 2).contiguous()

            # local_s2cnn_features = torch.zeros(batch_size, self.concat_feature_channels-3, self.num_centroids).to(point.device)
            local_s2cnn_features_list.append(local_s2cnn_features)

        local_s2cnn_features_list.append(new_xyz)
        x = torch.cat(local_s2cnn_features_list, dim=1)

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
        net = PN2S2CNNCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            num_centroids=cfg.MODEL.PNS2CNN.NUM_CENTROIDS,
            radius_list=cfg.MODEL.PNS2CNN.RADIUS_LIST,
            num_neighbours_list=cfg.MODEL.PNS2CNN.NUM_NEIGHBOURS_LIST,
            band_width_in_list=cfg.MODEL.PNS2CNN.BAND_WIDTH_IN_LIST,
            s2cnn_feature_channels_list=cfg.MODEL.PNS2CNN.FEATURE_CHANNELS_LIST,
            band_width_list=cfg.MODEL.PNS2CNN.BAND_WIDTH_LIST,
            k=cfg.MODEL.PNS2CNN.K,
            local_channels=cfg.MODEL.PNS2CNN.LOCAL_CHANNELS,
            global_channels=cfg.MODEL.PNS2CNN.GLOBAL_CHANNELS,
            dropout_prob=cfg.MODEL.PNS2CNN.DROPOUT_PROB,
            transform_xyz=cfg.MODEL.PNS2CNN.TRANSFORM_XYZ)

        loss_fn = PNS2CNNClsLoss(
            trans_reg_weight=cfg.MODEL.PNS2CNN.TRANS_REG_WEIGHT
        )
        metric_fn = Accuracy()

    else:
        raise NotImplementedError

    return net, loss_fn, metric_fn


if __name__ == "__main__":
    batch_size = 7
    in_channels = 3
    num_points = 1024
    num_classes = 40

    data = torch.rand(batch_size, in_channels, num_points)
    data = data.cuda()

    pns2cnn = PN2S2CNNCls(3, 40)
    pns2cnn = pns2cnn.cuda()

    out_dict = pns2cnn({"points": data})
    for k, v in out_dict.items():
        print("pns2cnn: ", k, v.size())
