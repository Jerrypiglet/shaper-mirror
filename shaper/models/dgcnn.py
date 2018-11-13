"""
DGCNN

References:
    @article{dgcnn,
      title={Dynamic Graph CNN for Learning on Point Clouds},
      author={Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon},
      journal={arXiv preprint arXiv:1801.07829},
      year={2018}
    }
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._dgcnn_utils import get_edge_feature
from ._utils import Conv1d, Conv2d, FC
from .pointnet import PointNetLocal, PointNetGlobal


class DGCNN_TNet(nn.Module):
    """
    DGCNN Transform Net
    Input:
        input_feature: tensor, (batch_size, in_channels, num_nodes)

    Returns:
        transform_matrix: tensor, (batch_size, out_channels, in_channels)
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 local_channels=(64, 128),
                 inter_channels=(1024,),
                 global_channels=(512, 256),
                 k=20, bn=True):
        super(DGCNN_TNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        in_channels *= 2
        self.k = k
        self.mlp_local = nn.ModuleList()
        for local_conv_channels in local_channels:
            self.mlp_local.append(Conv2d(in_channels, local_conv_channels, 1, bn=bn))
            in_channels = local_conv_channels

        self.mlp_inter = PointNetLocal(in_channels, inter_channels, bn=bn)
        self.mlp_global = PointNetGlobal(self.mlp_inter.out_channels, global_channels, bn=bn)
        self.linear = nn.Linear(self.mlp_global.out_channels, self.in_channels * out_channels, bias=True)

        self.init_weights()

    def forward(self, x):
        x = get_edge_feature(x, self.k)
        for local_conv in self.mlp_local:
            x = local_conv(x)
        x, _ = torch.max(x, -1)
        x = self.mlp_inter(x)
        x, _ = torch.max(x, -1)
        x = self.mlp_global(x)
        # print('TNet mlp_global output: ', x.size())
        x = self.linear(x)
        x = x.view(-1, self.out_channels, self.in_channels)
        I = torch.eye(self.out_channels, self.in_channels, device=x.device)
        x.add_(I)  # broadcast first dimension
        return x

    def init_weights(self):
        # set linear transform be 0
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


class DGCNN_GraphLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 graph_layer_channels=(64, 128, 256),
                 k=20, bn=True):
        super(DGCNN_GraphLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_conv_list = nn.ModuleList()
        for graph_channels in graph_layer_channels:
            self.graph_conv_list.append(Conv2d(2 * in_channels, graph_channels, 1, relu=True, bn=bn))
            in_channels = graph_channels
        self.final_conv = Conv1d(np.sum(graph_layer_channels), out_channels)
        self.k = k

    def forward(self, x):
        layer_feature_list = []
        for layer_conv in self.graph_conv_list:
            edge_feature = get_edge_feature(x, self.k)  # [N, C, H, W]
            x = layer_conv(edge_feature)
            x, _ = torch.max(x, -1)  # [N, C, H]
            layer_feature_list.append(x)
        x = torch.cat(tuple(layer_feature_list), 1)
        x = self.final_conv(x)
        return x


class DGCNN_Global(nn.ModuleList):
    def __init__(self, in_channels,
                 global_channels=(256, 128),
                 bn=True):
        super(DGCNN_Global, self).__init__()

        for ind, out_channels in enumerate(global_channels):
            self.append(FC(in_channels, out_channels, bn=bn))
            in_channels = out_channels
        self.out_channels = in_channels

    def forward(self, x):
        for module in self:
            x = module(x)
        return x


class DGCNN_Cls(nn.Module):
    def __init__(self, in_channels, out_channels, k=20,
                 graph_layer_channels=(64, 128, 256),
                 inter_layer_channels=256,
                 global_channels=(256, 128),
                 bn=True):
        super(DGCNN_Cls, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # input transform
        self.transform_input = DGCNN_TNet(in_channels, in_channels, bn=bn)
        self.graph_layer = DGCNN_GraphLayer(in_channels, inter_layer_channels, graph_layer_channels, k, bn=bn)
        self.mlp_global = DGCNN_Global(self.graph_layer.out_channels, global_channels, bn=bn)
        self.linear = nn.Linear(self.mlp_global.out_channels, out_channels, bias=False)

        self.init_weights()

    def forward(self, data_batch):
        end_points = {}
        x = data_batch["points"]
        trans_input = self.transform_input(x)
        x = torch.bmm(trans_input, x)
        end_points['trans_input'] = trans_input

        x = self.graph_layer(x)
        x, max_indices = torch.max(x, -1)
        end_points['key_point_inds'] = max_indices
        x = self.mlp_global(x)
        x = self.linear(x)
        preds = {
            'cls_logits': x
        }
        preds.update(end_points)

        return preds

    def init_weights(self):
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='linear')


class DGCNN_ClsLoss(nn.Module):
    def __init__(self, label_smoothing):
        super(DGCNN_ClsLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, preds, labels):
        cls_logits = preds["cls_logits"]
        cls_labels = labels["cls_labels"]
        if self.label_smoothing > 0:
            num_classes = cls_logits.size(-1)
            one_hot = torch.zeros_like(cls_logits).scatter(1, cls_labels.view(-1, 1), 1)
            smooth_one_hot = one_hot * (1 - self.label_smoothing) \
                      + torch.ones_like(cls_logits) * self.label_smoothing / num_classes
            log_prob = F.log_softmax(cls_logits, dim=1)
            loss = nn.KLDivLoss()
            cls_loss = loss(log_prob, smooth_one_hot)
        else:
            cls_loss = F.cross_entropy(cls_logits, cls_labels)
        loss_dict = {
            'cls_loss': cls_loss,
        }
        return loss_dict

class DGCNN_Metric(nn.Module):
    def forward(self, preds, labels):
        cls_logits = preds["cls_logits"]
        cls_labels = labels["cls_labels"]
        pred_labels = cls_logits.argmax(1)
        acc = pred_labels.eq(cls_labels).float().mean()

        metric_dict = {
            'acc': acc
        }
        return metric_dict

def build_dgcnn(cfg):
    if cfg.TASK == "classification":
        net = DGCNN_Cls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            k=cfg.MODEL.DGCNN.K,
            graph_layer_channels=cfg.MODEL.DGCNN.GRAPH_LAYER_CHANNELS,
            inter_layer_channels=cfg.MODEL.DGCNN.INTER_LAYER_CHANNELS,
            global_channels=cfg.MODEL.DGCNN.GLOBAL_CHANNELS
        )
        loss_fn = DGCNN_ClsLoss(cfg.MODEL.DGCNN.LABEL_SMOOTH)
        metric_fn = DGCNN_Metric()
    else:
        raise NotImplementedError()

    return net, loss_fn, metric_fn


if __name__ == "__main__":
    batch_size = 4
    in_channels = 3
    num_points = 1024
    num_classes = 10

    data = torch.rand(batch_size, in_channels, num_points)
    transform = DGCNN_TNet()
    out = transform(data)
    print('DGCNN_TNet: ', out.size())

    dgcnn = DGCNN_Cls(in_channels, num_classes)
    out, _ = dgcnn(data)
    print('dgcnn: ', out.size())
