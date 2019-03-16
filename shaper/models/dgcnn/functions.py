"""Helpers for DGCNN"""

import torch

from shaper.nn.functional import pdist
from shaper.models.dgcnn.gather_knn import gather_knn


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
        _, knn_inds = torch.topk(pdist, k + 1, largest=False, sorted=False)
        return knn_inds[..., 1:]
    else:
        _, knn_inds = torch.topk(pdist, k, largest=False, sorted=False)
        return knn_inds


def construct_edge_feature_index(feature, knn_inds):
    """Construct edge feature for each point (or regarded as a node)
    using advanced indexing

    Args:
        feature (torch.Tensor): point features, (batch_size, channels, num_nodes),
        knn_inds (torch.Tensor): indices of k-nearest neighbour, (batch_size, num_nodes, k)

    Returns:
        edge_feature: (batch_size, 2*channels, num_nodes, k)

    """
    batch_size, channels, num_nodes = feature.shape
    k = knn_inds.size(-1)

    feature_central = feature.unsqueeze(3).expand(batch_size, channels, num_nodes, k)
    batch_idx = torch.arange(batch_size).view(-1, 1, 1, 1)
    feature_idx = torch.arange(channels).view(1, -1, 1, 1)
    # (batch_size, channels, num_nodes, k)
    feature_neighbour = feature[batch_idx, feature_idx, knn_inds.unsqueeze(1)]
    edge_feature = torch.cat((feature_central, feature_neighbour - feature_central), 1)

    return edge_feature


def construct_edge_feature_gather(feature, knn_inds):
    """Construct edge feature for each point (or regarded as a node)
    using torch.gather

    Args:
        feature (torch.Tensor): point features, (batch_size, channels, num_nodes),
        knn_inds (torch.Tensor): indices of k-nearest neighbour, (batch_size, num_nodes, k)

    Returns:
        edge_feature: (batch_size, 2*channels, num_nodes, k)

    Notes:
        Pytorch Gather is 50x faster than advanced indexing, but needs 2x more memory.
        It is because it will allocate a tensor as large as expanded features during backward.

    """
    batch_size, channels, num_nodes = feature.shape
    k = knn_inds.size(-1)

    # CAUTION: torch.expand
    feature_central = feature.unsqueeze(3).expand(batch_size, channels, num_nodes, k)
    feature_expand = feature.unsqueeze(2).expand(batch_size, channels, num_nodes, num_nodes)
    knn_inds_expand = knn_inds.unsqueeze(1).expand(batch_size, channels, num_nodes, k)
    feature_neighbour = torch.gather(feature_expand, 3, knn_inds_expand)
    # (batch_size, 2 * channels, num_nodes, k)
    edge_feature = torch.cat((feature_central, feature_neighbour - feature_central), 1)

    return edge_feature


def construct_edge_feature(feature, knn_inds, use_centroids=True):
    """Construct edge feature for each point (or regarded as a node)
    using gather_knn

    Args:
        feature (torch.Tensor): point features, (batch_size, channels, num_nodes),
        knn_inds (torch.Tensor): indices of k-nearest neighbour, (batch_size, num_nodes, k)

    Returns:
        edge_feature: (batch_size, 2*channels, num_nodes, k)

    """
    batch_size, channels, num_nodes = feature.shape
    k = knn_inds.size(-1)

    # CAUTION: torch.expand
    feature_central = feature.unsqueeze(3).expand(batch_size, channels, num_nodes, k)
    feature_neighbour = gather_knn(feature, knn_inds)
    if use_centroids:
        # (batch_size, 2 * channels, num_nodes, k)
        edge_feature = torch.cat((feature_central, feature_neighbour - feature_central), 1)
    else:
        edge_feature = feature_neighbour - feature_central

    return edge_feature


def get_edge_feature(feature, k):
    """Get edge feature for point features

    Args:
        feature (torch.Tensor): (batch_size, channels, num_nodes)
        k (int): the number of nearest neighbours

    Returns:
        edge_feature (torch.Tensor): (batch_size, 2*num_dims, num_nodes, k)

    """
    with torch.no_grad():
        distance = pdist(feature)
        knn_inds = get_knn_inds(distance, k)
        # knn_inds = torch.ones(feature.size(0), feature.size(2), k, dtype=torch.int64, device=feature.device)

    edge_feature = construct_edge_feature(feature, knn_inds)
    # edge_feature = construct_edge_feature_gather(feature, knn_inds)
    # edge_feature = construct_edge_feature_index(feature, knn_inds)

    # edge_feature = torch.ones(feature.size(0), 2*feature.size(1), feature.size(2), k, device=feature.device)
    return edge_feature
