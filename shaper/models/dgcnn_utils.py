"""Helpers for DGCNN"""

import torch
from torch import nn
import torch.nn.functional as F

from shaper.nn.modules import SharedMLP


def pairwise_distance(features):
    """Compute pairwise distances of features.

    Args:
        features (torch.Tensor): (batch_size, channels, num_features)

    Returns:
        distance (torch.Tensor): (batch_size, num_features, num_features)

    Notes:
        This method returns square distances, and is optimized for lower memory and faster speed.
        Sqaure sum is more efficient than gather diagonal from inner product.

    """
    square_sum = torch.sum(features ** 2, 1, keepdim=True)
    square_sum = square_sum + square_sum.transpose(1, 2)
    distance = torch.baddbmm(square_sum, features.transpose(1, 2), features, alpha=-2.0)
    return distance


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
        _, knn_inds = torch.topk(pdist, k+1, largest=False)
        return knn_inds[..., 1:]
    else:
        _, knn_inds = torch.topk(pdist, k, largest=False)
        return knn_inds


def construct_edge_feature(features, knn_inds):
    """Construct edge feature for each point (or regarded as a node)

    Args:
        features (torch.Tensor): point features, (batch_size, channels, num_nodes),
        knn_inds (torch.Tensor): indices of k-nearest neighbour, (batch_size, num_nodes, k)

    Returns:
        edge_feature: (batch_size, 2*channels, num_nodes, k)

    Notes:
        Gather is more efficient than advance indexing

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


def get_edge_feature(features, k):
    """Get edge feature for point features

    Args:
        features (torch.Tensor): (batch_size, channels, num_nodes)
        k (int): the number of nearest neighbours

    Returns:
        edge_feature (torch.Tensor): (batch_size, 2*num_dims, num_nodes, k)

    """
    with torch.no_grad():
        pdist = pairwise_distance(features)
        knn_inds = get_knn_inds(pdist, k)

    edge_feature = construct_edge_feature(features, knn_inds)

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
            pdist = pairwise_distance(features)  # (batch_size, num_points, num_points)
            knn_inds = get_knn_inds(pdist, self.k)  # (batch_size, num_points, k)

        knn_inds_expand = knn_inds.unsqueeze(1).expand(batch_size, self.out_channels, num_points, self.k)
        edge_feature_expand = edge_feature.unsqueeze(2).expand(batch_size, self.out_channels, num_points, num_points)
        # (batch_size, out_channels, num_points, k)
        neighbour_feature = torch.gather(edge_feature_expand, 3, knn_inds_expand)
        # (batch_size, out_channels, num_points, k)
        edge_feature = (local_feature + edge_feature).unsqueeze(3) - neighbour_feature

        edge_feature = self.bn(edge_feature)
        edge_feature = F.relu(edge_feature, inplace=True)

        # max pooling over k neighbours
        edge_feature, _ = torch.max(edge_feature, 3)

        return edge_feature


if __name__ == '__main__':
    import time
    import numpy as np
    import scipy.spatial.distance as sdist

    batch_size = 16
    num_points = 1024
    channels = 64
    k = 20

    features = np.random.rand(batch_size, channels, num_points)
    features_tensor = torch.from_numpy(features).cuda()

    # check pairwise distance
    pdist = np.stack([sdist.squareform(np.square(sdist.pdist(feat.T))) for feat in features])

    with torch.no_grad():
        for warmup in range(5):
            pdist_tensor = pairwise_distance(features_tensor)
        end = time.time()
        for _ in range(50):
            pdist_tensor = pairwise_distance(features_tensor)
        print("pairwise distance", (time.time() - end) / 50, torch.cuda.max_memory_cached() / 1024 ** 2)

    print(np.allclose(pdist, pdist_tensor.cpu().numpy()))

    # check construct edge feature
    torch.cuda.empty_cache()
    with torch.no_grad():
        for warmup in range(5):
            pdist_tensor = pairwise_distance(features_tensor)

        # knn_inds = torch.arange(k).view(1, 1, -1).expand(batch_size, num_points, k).cuda()
        _, knn_inds = torch.topk(pdist_tensor, k, largest=False)

        end = time.time()
        for _ in range(5):
            edge_feature = construct_edge_feature(features_tensor, knn_inds)
        print("construct edge feature", (time.time() - end) / 5, torch.cuda.max_memory_cached() / 1024 ** 2)

    # check module speed
    features = np.random.rand(batch_size, channels, num_points).astype(np.float32)
    features_tensor = torch.from_numpy(features)

    torch.cuda.empty_cache()
    edge_conv = EdgeConvBlockV2(channels, 64, k)
    end = time.time()
    with torch.no_grad():
        for _ in range(5):
            edge_feature = edge_conv(features_tensor)
        print("EdgeConvBlockV2", (time.time() - end) / 5, torch.cuda.max_memory_cached() / 1024 ** 2)

    torch.cuda.empty_cache()
    edge_conv = EdgeConvBlock(channels, [64], k)
    end = time.time()
    with torch.no_grad():
        for _ in range(5):
            edge_feature = edge_conv(features_tensor)
        print("EdgeConvBlock", (time.time() - end) / 5, torch.cuda.max_memory_cached() / 1024 ** 2)
