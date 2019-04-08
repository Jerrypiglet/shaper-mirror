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
from shaper.nn.init import set_bn, xavier_uniform


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
class DGCNNTwoBranch(nn.Module):
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
                 num_global_output=0,
                 num_mask_output=0,
                 edge_conv_channels=((64, 64), (64, 64), (64, 64)),
                 inter_channels=1024,
                 global_channels=(256, 256, 128),
                 k=20,
                 dropout_prob=0.4,
                 with_transform=True,
                 use_bn=True,
                 use_gn=False):
        super(DGCNNTwoBranch, self).__init__()

        self.in_channels = in_channels
        self.num_mask_output = num_mask_output
        self.num_global_output = num_global_output
        self.k = k
        self.with_transform = with_transform
        self.num_gpu = torch.cuda.device_count()

        # input transform
        if self.with_transform:
            self.transform_input = TNet(in_channels, in_channels, k=k, use_bn=use_bn, use_gn=use_gn)

        self.mlp_edge_conv = nn.ModuleList()
        for out in edge_conv_channels:
            self.mlp_edge_conv.append(EdgeConvBlock(in_channels, out, k, use_bn=use_bn, use_gn=use_gn))
            in_channels = out[-1]

        out_channel = edge_conv_channels[0][0]

        mlplocal_input = sum([item[-1] for item in edge_conv_channels])
        self.mlp_local = Conv1d(mlplocal_input, inter_channels, 1, bn=use_bn, gn=use_gn)

        if num_global_output > 0:
            self.mlp_global = MLP(inter_channels, global_channels, dropout=dropout_prob, bn=use_bn, gn=use_gn)
            self.global_output = nn.Linear(global_channels[-1], num_global_output, bias=True)

        if num_mask_output > 0:
            mlp_in_channels = inter_channels + edge_conv_channels[-1][-1] + sum([item[-1] for item in edge_conv_channels])
            mlp_in_channels = inter_channels +  sum([item[-1] for item in edge_conv_channels])
            self.mlp_seg = SharedMLP(mlp_in_channels, global_channels[:-1], dropout=dropout_prob, bn=use_bn, gn=use_gn)
            self.conv_seg = Conv1d(global_channels[-2], global_channels[-1], 1, bn=use_bn, gn=use_gn)
            self.mask_output = nn.Conv1d(global_channels[-1], num_mask_output, 1, bias=True)

        self.init_weights()
        set_bn(self, momentum=0.01)

    def forward(self, data_batch):
        end_points = {}
        x = data_batch["points"]

        num_point = x.shape[2]
        batch_size = x.size()[0]
        preds={}

        if self.with_transform:
            trans_input = self.transform_input(x)
            x = torch.bmm(trans_input, x)
            end_points['trans_input'] = trans_input

        # edge convolution for point cloud
        features = []
        for edge_conv in self.mlp_edge_conv:
            x = edge_conv(x)
            features.append(x)

        x = torch.cat(features, dim=1)

        # local mlp
        x = self.mlp_local(x)
        x, max_indice = torch.max(x, 2)


        if self.num_global_output>0:
            y = self.mlp_global(x)
            preds['global_output'] = self.global_output(y)

        if self.num_mask_output > 0 :

            end_points['key_point_inds'] = max_indice

            x=x.unsqueeze(2).expand(-1,-1,num_point)
            cat_features = torch.cat(features, dim=1)
            x = torch.cat([x, cat_features],dim=1)

            # mlp_seg & conv_seg
            x = self.mlp_seg(x)
            x = self.conv_seg(x)
            mask_output = self.mask_output(x)
            preds['mask_output'] = mask_output
        preds.update(end_points)

        return preds

    def init_weights(self):

        for edge_conv in self.mlp_edge_conv:
            edge_conv.init_weights(xavier_uniform)
        self.mlp_local.init_weights(xavier_uniform)
        self.mlp_global.init_weights(xavier_uniform)
        self.mlp_seg.init_weights(xavier_uniform)
        self.conv_seg.init_weights(xavier_uniform)
        nn.init.xavier_uniform_(self.mask_output.weight)
        nn.init.zeros_(self.mask_output.bias)
        nn.init.xavier_uniform(self.global_output.weight)
        nn.init.zeros_(self.global_output.bias)
        set_bn(self, momentum=0.01)


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
