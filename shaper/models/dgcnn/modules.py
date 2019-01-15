import torch
from torch import nn
import torch.nn.functional as F

from shaper.nn import SharedMLP
from shaper.nn.init import init_bn
from .functions import pdist, get_knn_inds, get_edge_feature, gather_knn


class EdgeConvBlock(nn.Module):
    """EdgeConv Block

    Structure: point features -> [get_edge_feature] -> edge features -> [MLP(2d)]
    -> [MaxPool] -> point features

    """

    def __init__(self, in_channels, out_channels, k):
        super(EdgeConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        self.mlp = SharedMLP(2 * in_channels, out_channels, ndim=2)

    def forward(self, x):
        x = get_edge_feature(x, self.k)
        x = self.mlp(x)
        x, _ = torch.max(x, 3)
        return x

    def init_weights(self, init_fn=None):
        self.mlp.init_weights(init_fn)


class EdgeConvBlockV2(nn.Module):
    """EdgeConv Block V2

    This implementation costs theoretically k times less computation than EdgeConvBlock.
    However, it is slower than original EdgeConvBlock when using gpu.
    Using cpu, it is slightly faster. The memory usage is also slightly smaller.

    """

    def __init__(self, in_channels, out_channels, k):
        super(EdgeConvBlockV2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, feature):
        batch_size, _, num_points = feature.shape

        local_feature = self.conv1(feature)  # (batch_size, out_channels, num_points)
        edge_feature = self.conv2(feature)  # (batch_size, out_channels, num_points)

        # calculate k-nn on raw feature
        with torch.no_grad():
            distance = pdist(feature)  # (batch_size, num_points, num_points)
            knn_inds = get_knn_inds(distance, self.k)  # (batch_size, num_points, k)

        # pytorch gather
        # knn_inds_expand = knn_inds.unsqueeze(1).expand(batch_size, self.out_channels, num_points, self.k)
        # edge_feature_expand = edge_feature.unsqueeze(2).expand(batch_size, self.out_channels, num_points, num_points)
        # # (batch_size, out_channels, num_points, k)
        # neighbour_feature = torch.gather(edge_feature_expand, 3, knn_inds_expand)

        # improved gather
        neighbour_feature = gather_knn(edge_feature, knn_inds)
        # (batch_size, out_channels, num_points, k)
        edge_feature = (local_feature + edge_feature).unsqueeze(3) - neighbour_feature

        edge_feature = self.bn(edge_feature)
        edge_feature = F.relu(edge_feature, inplace=True)

        # max pooling over k neighbours
        edge_feature, _ = torch.max(edge_feature, 3)

        return edge_feature

    def init_weights(self, init_fn=None):
        if init_fn is not None:
            init_fn(self.conv1)
            init_fn(self.conv2)
        init_bn(self.bn)
