import numpy as np

import torch
import torch.nn as nn

from _utils import Conv1dBlock, Conv2dBlock, LinearBlock
from pointnet import PointNetLocal, PointNetGlobal


def cal_pairwise_dist(input_feature):
    """
    Compute pairwise distance of input_feature.
    Args:
        input_feature: tensor (batch_size, num_dims, num_nodes)

    Returns:
        pairwise_distance: tensor (batch_size, num_nodes, num_nodes)
    """
    batch_size, num_dims, num_nodes = list(input_feature.size())
    # feature = input_feature.squeeze()
    # if batch_size == 1:
    #     feature.unqueeze_(0)
    feature_transpose = input_feature.transpose(1, 2)
    feature_inner = torch.matmul(feature_transpose, input_feature)  # (batch_size, num_nodes, num_nodes)
    feature_inner = -2 * feature_inner
    feature_square_sum = torch.sum(input_feature ** 2, 1, keepdim=True)
    feature_transpose_square_sum = feature_square_sum.transpose(1, 2)

    pairwise_dist = feature_square_sum + feature_inner + feature_transpose_square_sum

    return pairwise_dist


def get_knn_inds(pairwise_dist, k=20):
    """
    Get k nearest neighbour index based on the pairwise_distance.
    Args:
        pairwise_dist: tensor (batch_size, num_nodes, num_nodes)
        k: int

    Returns:
        knn_inds: (batch_size, num_nodes, k)
    """
    _, knn_inds = torch.topk(pairwise_dist, k, largest=False)
    return knn_inds


def construct_edge_feature(feature, knn_inds):
    """
    Construct edge feature for each point
    Args:
        feature: (batch_size, num_dims, num_nodes)
        knn_inds: (batch_size, num_nodes, k)

    Returns:
        edge_features: (batch_size, 2*num_dims, num_nodes, k)
    """
    batch_size, num_dims, num_nodes = list(feature.size())
    k = list(knn_inds.size())[-1]

    feature_central = feature.unsqueeze(-1).repeat(1, 1, 1, k)
    feature_tile = feature.unsqueeze(-1).repeat(1, 1, 1, num_nodes)
    knn_inds = knn_inds.unsqueeze(1).repeat(1, num_dims, 1, 1)
    feature_neighbour = torch.gather(feature_tile, -1, knn_inds)  # (batch_size, num_dims, num_nodes, k)

    edge_feature = torch.cat((feature_central, feature_neighbour - feature_central), 1)

    return edge_feature


def get_edge_feature(input_feature, k):
    """
    Get edge feature for input_feature
    Args:
        input_feature: (batch_size, num_dims, num_nodes)
        k:int, # of nearest neighbours

    Returns:
        edge_feature: (batch_size, 2*num_dims, num_nodes, k)
    """
    pairwise_dist = cal_pairwise_dist(input_feature)
    knn_inds = get_knn_inds(pairwise_dist, k)
    edge_feature = construct_edge_feature(input_feature, knn_inds)

    return edge_feature


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
        self.mlp_local = []
        for local_conv_channels in local_channels:
            self.mlp_local.append(Conv2dBlock(in_channels, local_conv_channels, 1, bn=bn))
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
        self.graph_conv_list = []
        for graph_channels in graph_layer_channels:
            self.graph_conv_list.append(Conv2dBlock(2 * in_channels, graph_channels, 1, relu=True, bn=bn))
            in_channels = graph_channels
        self.final_conv = Conv1dBlock(np.sum(graph_layer_channels), out_channels)
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
            self.append(LinearBlock(in_channels, out_channels, bn=bn))
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

    def forward(self, x):
        end_points = {}
        trans_input = self.transform_input(x)
        x = torch.bmm(trans_input, x)
        end_points['trans_input'] = trans_input

        x = self.graph_layer(x)
        x, max_indices = torch.max(x, -1)
        end_points['key_point_inds'] = max_indices
        x = self.mlp_global(x)
        x = self.linear(x)

        return x, end_points

    def init_weights(self):
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='linear')


if __name__ == "__main__":
    batch_size = 4
    in_channels = 3
    num_points = 1024
    num_classes = 10
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = torch.rand(batch_size, in_channels, num_points)
    transform = DGCNN_TNet()
    out = transform(data)
    print('DGCNN_TNet: ', out.size())

    dgcnn = DGCNN_Cls(in_channels, num_classes)
    out, _ = dgcnn(data)
    print('dgcnn: ', out.size())
