"""Helpers for DGCNN"""

import torch
from torch import nn
import torch.nn.functional as F

from shaper.nn.modules import SharedMLP
from shaper.nn.functional import pdist

from shaper.models.dgcnn_modules.gather_knn import gather_knn


def get_knn_inds(pdist, k=20, remove=False):
    """Get k nearest neighbour index based on the pairwise_distance.

    Args:
        pdist (torch.Tensor): tensor (batch_size, num_nodes, num_nodes)
        k (int): the number of nearest neighbour
        remove (bool): whether to remove itself

    Returns:
        knn_inds (torch.Tensor): (batch_size, num_nodes, k)

    """
    if remove:
        _, knn_inds = torch.topk(pdist, k + 1, largest=False)
        return knn_inds[..., 1:]
    else:
        _, knn_inds = torch.topk(pdist, k, largest=False)
        return knn_inds


def construct_edge_feature_index(features, knn_inds):
    """Construct edge feature for each point (or regarded as a node)
    using advanced indexing

    Args:
        features (torch.Tensor): point features, (batch_size, channels, num_nodes),
        knn_inds (torch.Tensor): indices of k-nearest neighbour, (batch_size, num_nodes, k)

    Returns:
        edge_feature: (batch_size, 2*channels, num_nodes, k)

    """
    batch_size, channels, num_nodes = features.shape
    k = knn_inds.size(-1)

    feature_central = features.unsqueeze(3).expand(batch_size, channels, num_nodes, k)
    batch_idx = torch.arange(batch_size).view(-1, 1, 1, 1)
    feature_idx = torch.arange(channels).view(1, -1, 1, 1)
    # (batch_size, channels, num_nodes, k)
    feature_neighbour = features[batch_idx, feature_idx, knn_inds.unsqueeze(1)]
    edge_feature = torch.cat((feature_central, feature_neighbour - feature_central), 1)

    return edge_feature


def construct_edge_feature_gather(features, knn_inds):
    """Construct edge feature for each point (or regarded as a node)
    using torch.gather

    Args:
        features (torch.Tensor): point features, (batch_size, channels, num_nodes),
        knn_inds (torch.Tensor): indices of k-nearest neighbour, (batch_size, num_nodes, k)

    Returns:
        edge_feature: (batch_size, 2*channels, num_nodes, k)

    Notes:
        Pytorch Gather is 50x faster than advanced indexing, but needs 2x more memory.
        It is because it will allocate a tensor as large as expanded features during backward.

    """
    batch_size, channels, num_nodes = features.shape
    k = knn_inds.size(-1)

    # CAUTION: torch.expand
    feature_central = features.unsqueeze(3).expand(batch_size, channels, num_nodes, k)
    feature_expand = features.unsqueeze(2).expand(batch_size, channels, num_nodes, num_nodes)
    knn_inds_expand = knn_inds.unsqueeze(1).expand(batch_size, channels, num_nodes, k)
    feature_neighbour = torch.gather(feature_expand, 3, knn_inds_expand)
    # (batch_size, 2 * channels, num_nodes, k)
    edge_feature = torch.cat((feature_central, feature_neighbour - feature_central), 1)

    return edge_feature


def construct_edge_feature(features, knn_inds):
    """Construct edge feature for each point (or regarded as a node)
    using gather_knn

    Args:
        features (torch.Tensor): point features, (batch_size, channels, num_nodes),
        knn_inds (torch.Tensor): indices of k-nearest neighbour, (batch_size, num_nodes, k)

    Returns:
        edge_feature: (batch_size, 2*channels, num_nodes, k)

    """
    batch_size, channels, num_nodes = features.shape
    k = knn_inds.size(-1)

    # CAUTION: torch.expand
    feature_central = features.unsqueeze(3).expand(batch_size, channels, num_nodes, k)
    feature_neighbour = gather_knn(features, knn_inds)
    # (batch_size, 2 * channels, num_nodes, k)
    edge_feature = torch.cat((feature_central, feature_neighbour - feature_central), 1)

    return edge_feature


def get_edge_feature(features, k):
    """Get edge feature for point features

    Args:
        features (torch.Tensor): (batch_size, channels, num_nodes)
        k (int): the number of nearest neighbours

    Returns:
        edge_feature (torch.Tensor): (batch_size, 2*num_dims, num_nodes, k)

    """
    with torch.no_grad():
        distance = pdist(features)
        knn_inds = get_knn_inds(distance, k)
        # knn_inds = torch.ones(features.size(0), features.size(2), k, dtype=torch.int64, device=features.device)

    edge_feature = construct_edge_feature(features, knn_inds)
    # edge_feature = construct_edge_feature_gather(features, knn_inds)
    # edge_feature = construct_edge_feature_index(features, knn_inds)

    # edge_feature = torch.ones(features.size(0), 2*features.size(1), features.size(2), k, device=features.device)
    return edge_feature


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

    def forward(self, features):
        batch_size, _, num_points = features.shape

        local_feature = self.conv1(features)  # (batch_size, out_channels, num_points)
        edge_feature = self.conv2(features)  # (batch_size, out_channels, num_points)

        # calculate k-nn on raw features
        with torch.no_grad():
            distance = pdist(features)  # (batch_size, num_points, num_points)
            knn_inds = get_knn_inds(distance, self.k)  # (batch_size, num_points, k)

        # gather
        # knn_inds_expand = knn_inds.unsqueeze(1).expand(batch_size, self.out_channels, num_points, self.k)
        # edge_feature_expand = edge_feature.unsqueeze(2).expand(batch_size, self.out_channels, num_points, num_points)
        # # (batch_size, out_channels, num_points, k)
        # neighbour_feature = torch.gather(edge_feature_expand, 3, knn_inds_expand)
        neighbour_feature = gather_knn(edge_feature, knn_inds)
        # (batch_size, out_channels, num_points, k)
        edge_feature = (local_feature + edge_feature).unsqueeze(3) - neighbour_feature

        edge_feature = self.bn(edge_feature)
        edge_feature = F.relu(edge_feature, inplace=True)

        # max pooling over k neighbours
        edge_feature, _ = torch.max(edge_feature, 3)

        return edge_feature
