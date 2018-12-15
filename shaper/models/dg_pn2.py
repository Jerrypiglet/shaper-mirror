"""
Dynamic Graph + Local PointNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.nn.functional import smooth_cross_entropy
from shaper.models.pn2_modules.pointnet2_modules import PointnetSAModuleMSG
from shaper.models.dgcnn import TNet
from shaper.models.dgcnn import DGCNNFeature
from shaper.models.metric import Accuracy
from shaper.nn import SharedMLP, MLP
from shaper.models.dgcnn_modules.dgcnn_utils import get_edge_feature


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


class DGPN2Cls(nn.Module):
    """Dynamic Graph + Local PointNetMSG for classification

    Structure: input -> [PointNetSAModuleMSG] (-> [TNet] -> transform_group_center)
    -> [Concat] -> [DGCNN] -> logits

    Attributes:
        transform_xyz： whether to transform group center coordinates
    """

    def __init__(self, in_channels, out_channels,
                 num_points=256, radius_list=(0.2,), num_samples_list=(64,),
                 group_mlps_list=((64, 64, 128),),  # local pointnet paras
                 edge_conv_channels=(128, 256, 512), inter_channels=128,  # dynamic graph paras
                 global_channels=(512, 256), k=20, transform_xyz=True, drop_prob=0.5):
        super(DGPN2Cls, self).__init__()

        local_sa_scale_num = len(radius_list)
        assert len(num_samples_list) == local_sa_scale_num
        assert len(group_mlps_list) == local_sa_scale_num
        self.transform_xyz = transform_xyz
        self.k = k
        self.out_channels = out_channels

        feature_channels = in_channels - 3
        group_sa_out_channels = 0
        group_sa_mlps = []
        for _ in group_mlps_list:
            _ = list(_)
            _.insert(0, feature_channels)
            group_sa_mlps.append(_)
            group_sa_out_channels += _[-1]

        self.SA_modules = PointnetSAModuleMSG(npoint=num_points,
                                              radii=list(radius_list),
                                              nsamples=list(num_samples_list),
                                              mlps=group_sa_mlps, use_xyz=True)

        concat_feature_channels = group_sa_out_channels + 3  # features+xyz

        if self.transform_xyz:
            self.transform_input = TNet(concat_feature_channels, 3, k=k)

        self.dgcnn = DGCNNFeature(in_channels=concat_feature_channels,
                                  edge_conv_channels=edge_conv_channels,
                                  inter_channels=inter_channels,
                                  global_channels=global_channels, k=k,
                                  dropout_prob=drop_prob, with_transform=False)

        self.classifier = nn.Linear(self.dgcnn.out_channels, self.out_channels, bias=True)

        self.init_weights()

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, data_batch):
        end_points = {}
        pointcloud = data_batch["points"]
        pointcloud = pointcloud.transpose(1, 2)
        xyz, features = self._break_up_pc(pointcloud)
        xyz, features = self.SA_modules(xyz, features)
        xyz = xyz.transpose(1, 2)  # [b, 3, np]

        x = torch.cat((features, xyz), 1)  # [b, fc+3, np]

        if self.transform_xyz:
            trans_matrix_xyz = self.transform_input(x)
            xyz = torch.bmm(trans_matrix_xyz, xyz)
            end_points["trans_matrix_xyz"] = trans_matrix_xyz
            x = torch.cat((features, xyz), 1)  # [b, fc+3, np]

        x, end_points = self.dgcnn(x, end_points)
        x = self.classifier(x)

        preds = {
            'cls_logits': x
        }
        preds.update(end_points)

        return preds

    def init_weights(self):
        nn.init.kaiming_uniform_(self.classifier.weight, nonlinearity='linear')
        nn.init.zeros_(self.classifier.bias)


class DGPN2ClsLoss(nn.Module):
    def __init__(self, label_smoothing, trans_reg_weight):
        super(DGPN2ClsLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.trans_reg_weight = trans_reg_weight

    def forward(self, preds, labels):
        cls_logits = preds["cls_logits"]
        cls_labels = labels["cls_labels"]
        if self.label_smoothing > 0:
            cls_loss = smooth_cross_entropy(cls_logits, cls_labels, self.label_smoothing)
        else:
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


def build_dgpn2(cfg):
    if cfg.TASK == "classification":
        net = DGPN2Cls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            num_points=cfg.MODEL.DGPN2.NUM_POINTS,
            radius_list=cfg.MODEL.DGPN2.RADIUS_LIST,
            num_samples_list=cfg.MODEL.DGPN2.NUM_SAMPLES,
            group_mlps_list=cfg.MODEL.DGPN2.GROUP_MLPS,
            edge_conv_channels=cfg.MODEL.DGPN2.EDGE_CONV_CHANNELS,
            inter_channels=cfg.MODEL.DGPN2.INTER_CHANNELS,
            global_channels=cfg.MODEL.DGPN2.GLOBAL_CHANNELS,
            k=cfg.MODEL.DGPN2.K,
            transform_xyz=cfg.MODEL.DGPN2.TRANSFORM_XYZ,
            drop_prob=cfg.MODEL.DGPN2.DROP_PROB
        )

        loss_fn = DGPN2ClsLoss(
            label_smoothing=cfg.MODEL.DGPN2.LABEL_SMOOTH,
            trans_reg_weight=cfg.MODEL.DGPN2.TRANS_REG_WEIGHT
        )
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

    dgpn2 = DGPN2Cls(3, 40)
    dgpn2.cuda()
    out_dict = dgpn2({"points": data})
    for k, v in out_dict.items():
        print('dgpn2:', k, v.size())
