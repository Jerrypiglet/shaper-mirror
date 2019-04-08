"""DGCNN
References:
    @article{dgcnn,
      title={Dynamic Graph CNN for Learning on Point Clouds},
      author={Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon},
      journal={arXiv preprint arXiv:1801.07829},
      year={2018}
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.nn.functional import smooth_cross_entropy
from shaper.nn import MLP, SharedMLP, Conv1d, Conv2d
from shaper.models.dgcnn.functions import get_edge_feature
from shaper.models.dgcnn.modules import EdgeConvBlock
from shaper.nn.init import set_bn


class TNet(nn.Module):
    """Transformation Network for DGCNN

    Structure: input -> [EdgeFeature] -> [EdgeConv]s -> [EdgePool] -> features -> [MLP] -> local features
    -> [MaxPool] -> global features -> [MLP] -> [Linear] -> logits

    Args:
        conv_channels (tuple of int): the numbers of channels of edge convolution layers
        k: the number of neareast neighbours for edge feature extractor

    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 conv_channels=(64, 128),
                 local_channels=(1024,),
                 global_channels=(512, 256),
                 k=20,
                 use_bn=True,
                 use_gn=False):
        super(TNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        self.edge_conv = SharedMLP(2 * in_channels, conv_channels, ndim=2, bn=use_bn, gn=use_gn)
        self.mlp_local = SharedMLP(conv_channels[-1], local_channels, bn=use_bn, gn=use_gn)
        self.mlp_global = MLP(local_channels[-1], global_channels, bn=use_bn, gn=use_gn)

        self.linear = nn.Linear(global_channels[-1], self.in_channels * out_channels, bias=True)

        self.init_weights()

    def forward(self, x):
        """TNet forward

        Args:
            x (torch.Tensor): (batch_size, in_channels, num_points)

        Returns:
            torch.Tensor: (batch_size, out_channels, in_channels)

        """
        x = get_edge_feature(x, self.k)  # (batch_size, 2 * in_channels, num_points, k)
        x = self.edge_conv(x)
        x, _ = torch.max(x, 3)  # (batch_size, edge_channels[-1], num_points)
        x = self.mlp_local(x)
        x, _ = torch.max(x, 2)  # (batch_size, local_channels[-1], num_points)
        x = self.mlp_global(x)
        x = self.linear(x)
        x = x.view(-1, self.out_channels, self.in_channels)
        I = torch.eye(self.out_channels, self.in_channels, device=x.device)
        x = x.add(I)  # broadcast first dimension
        return x

    def init_weights(self):
        # set linear transform be 0
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


# -----------------------------------------------------------------------------
# DGCNN for part segmentation
# -----------------------------------------------------------------------------
class DGCNNPartSeg(nn.Module):
    """DGCNN for part segmentation
       Structure: (-> [TNet] -> transform_input) -> [EdgeConvBlock]s -> [Concat EdgeConvBlock features]]
       -> [local MLP] -> [add classification label info] -> [Concat Features] -> [mlp seg] -> [conv seg]
       -> [seg logit] -> logits

       [EdgeConvBlock]: in_feature -> [EdgeFeature] -> [EdgeConv] -> [EdgePool] -> out_features

       Args:
           in_channels: (int) dimension of input layer
           out_channels: (int) dimension of output layer
           num_seg_class: (int) number of segmentation class [shapenet: 50]
               edge_conv_channels: (tuple of int) numbers of channels of edge convolution layers
           inter_channels: (int) number of channels of intermediate features before MaxPool
               k: (int) number of nearest neighbours for edge feature extractor
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_class,
                 num_seg_class,
                 edge_conv_channels=((64, 64), (64, 64), (64, 64)),
                 inter_channels=1024,
                 global_channels=(256, 256, 128),
                 k=20,
                 dropout_prob=0.4,
                 with_transform=True,
                 use_bn=True,
                 use_gn=False):
        super(DGCNNPartSeg, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.with_transform = with_transform
        self.num_gpu = torch.cuda.device_count()
        self.num_class = num_class

        # input transform
        if self.with_transform:
            self.transform_input = TNet(in_channels, in_channels, k=k, use_bn=use_bn, use_gn=use_gn)

        self.mlp_edge_conv = nn.ModuleList()
        for out in edge_conv_channels:
            self.mlp_edge_conv.append(EdgeConvBlock(in_channels, out, k, use_bn=use_bn, use_gn=use_gn))
            in_channels = out[-1]

        out_channel = edge_conv_channels[0][0]
        self.lable_conv = Conv2d(num_class, out_channel, [1, 1], bn=use_bn, gn=use_gn)

        mlplocal_input = sum([item[-1] for item in edge_conv_channels])
        self.mlp_local = Conv1d(mlplocal_input, inter_channels, 1, bn=use_bn, gn=use_gn)

        mlp_in_channels = inter_channels + edge_conv_channels[-1][-1] + sum([item[-1] for item in edge_conv_channels])
        self.mlp_seg = SharedMLP(mlp_in_channels, global_channels[:-1], dropout=dropout_prob, bn=use_bn, gn=use_gn)
        self.conv_seg = Conv1d(global_channels[-2], global_channels[-1], 1, bn=use_bn, gn=use_gn)
        self.seg_logit = nn.Conv1d(global_channels[-1], num_seg_class, 1, bias=True)

        self.init_weights()
        set_bn(self, momentum=0.01)

    def forward(self, data_batch):
        end_points = {}
        x = data_batch["points"]
        cls_label = data_batch["cls_label"]

        num_point = x.shape[2]
        batch_size = cls_label.size()[0]
        num_classes = self.num_class

        if self.with_transform:
            trans_input = self.transform_input(x)
            x = torch.bmm(trans_input, x)
            end_points['trans_input'] = trans_input

        # edge convolution for point cloud
        features = []
        for edge_conv in self.mlp_edge_conv:
            x = edge_conv(x)
            features.append(x)
        print (x.shape)
        exit(0)

        x = torch.cat(features, dim=1)

        # local mlp
        x = self.mlp_local(x)
        x, max_indice = torch.max(x, 2)

        end_points['key_point_inds'] = max_indice

        # use info from classification label
        with torch.no_grad():
            I = torch.eye(16, dtype=x.dtype, device=x.device)
            one_hot = I[cls_label]
            one_hot_expand = one_hot.view(batch_size, num_classes, 1, 1)

        one_hot_expand = self.lable_conv(one_hot_expand)

        # concatenate information from point cloud and label
        one_hot_expand = one_hot_expand.view(batch_size, -1)
        out_max = torch.cat([x, one_hot_expand], dim=1)
        out_max = out_max.unsqueeze(2).expand(-1, -1, num_point)

        cat_features = torch.cat(features, dim=1)
        x = torch.cat([out_max, cat_features], dim=1)

        # mlp_seg & conv_seg
        x = self.mlp_seg(x)
        x = self.conv_seg(x)
        seg_logit = self.seg_logit(x)
        preds = {
            'seg_logit': seg_logit
        }
        preds.update(end_points)

        return preds

    def init_weights(self):
        nn.init.xavier_uniform_(self.seg_logit.weight)
        nn.init.zeros_(self.seg_logit.bias)


class DGCNNPartSegLoss(nn.Module):
    """DGCNN part segmentation loss with optional regularization loss"""

    def __init__(self, reg_weight, cls_loss_weight, seg_loss_weight):
        super(DGCNNPartSegLoss, self).__init__()
        self.reg_weight = reg_weight
        self.cls_loss_weight = cls_loss_weight
        self.seg_loss_weight = seg_loss_weight
        assert self.seg_loss_weight >= 0.0

    def forward(self, preds, labels):
        seg_logit = preds["seg_logit"]
        seg_label = labels["seg_label"]
        seg_loss = F.cross_entropy(seg_logit, seg_label)
        loss_dict = {
            "seg_loss": seg_loss * self.seg_loss_weight,
        }

        if self.cls_loss_weight > 0.0:
            cls_logit = preds["cls_logit"]
            cls_label = labels["cls_label"]
            cls_loss = F.cross_entropy(cls_logit, cls_label)
            loss_dict["cls_loss"] = cls_loss

        # regularization over transform matrix
        if self.reg_weight > 0.0:
            trans_feature = preds["trans_input"]
            trans_norm = torch.bmm(trans_feature.transpose(2, 1), trans_feature)  # [in, in]
            I = torch.eye(trans_norm.size(2), dtype=trans_norm.dtype, device=trans_norm.device)
            reg_loss = F.mse_loss(trans_norm, I.unsqueeze(0).expand_as(trans_norm), reduction="sum")
            loss_dict["reg_loss"] = reg_loss * (0.5 * self.reg_weight / trans_norm.size(0))
        return loss_dict


if __name__ == "__main__":
    batch_size = 4
    in_channels = 3
    num_points = 1024
    num_classes = 40

    data = torch.rand(batch_size, in_channels, num_points)
    transform = TNet()
    out = transform(data)
    print('TNet: ', out.size())

    dgcnn = DGCNNPartSeg(in_channels, num_classes, 16, 40, with_transform=False)
    out_dict = dgcnn({"points": data})
    for k, v in out_dict.items():
        print('DGCNN:', k, v.shape)
