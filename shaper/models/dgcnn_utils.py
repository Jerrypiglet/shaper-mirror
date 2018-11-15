"""Helpers for DGCNN"""

import torch


def pairwise_distance(features):
    """Compute pairwise distances of features.

    Args:
        features (torch.Tensor): (batch_size, channels, num_features)

    Returns:
        pairwise_distance (torch.Tensor): (batch_size, num_features, num_features)

    Notes:
        This method returns square distances.

    """
    # (batch_size, num_features, num_features)
    inner_product = torch.bmm(features.transpose(1, 2), features)

    batch_idx = torch.arange(features.size(0)).view(-1, 1)
    diag_idx = torch.arange(features.size(2)).view(1, -1)
    # advance indexing, (batch_size, num_features)
    norm = inner_product[batch_idx, diag_idx, diag_idx]

    distance = norm.unsqueeze(1) + norm.unsqueeze(2) - (2 * inner_product)

    return distance


def construct_edge_feature(features, knn_inds):
    """Construct edge feature for each point

    Args:
        features (torch.Tensor): (batch_size, channels, num_nodes)
        knn_inds (torch.Tensor): (batch_size, num_nodes, k)

    Returns:
        edge_features: (batch_size, 2*channels, num_nodes, k)

    """
    batch_size, channels, num_nodes = features.shape
    k = knn_inds.size(-1)

    feature_central = features.unsqueeze(3).expand(batch_size, channels, num_nodes, k)
    batch_idx = torch.arange(batch_size).view(-1, 1, 1, 1)
    feature_idx = torch.arange(channels).view(1, -1, 1, 1)
    # (batch_size, channels, num_nodes, k)
    feature_neighbour = features[batch_idx, feature_idx, knn_inds.unsqueeze(1)]
    # (batch_size, 2 * channels, num_nodes, k)
    edge_feature = torch.cat((feature_central, feature_neighbour - feature_central), 1)

    return edge_feature


def get_edge_feature(features, k):
    """Get edge feature for input_feature

    Args:
        features (torch.Tensor): (batch_size, channels, num_nodes)
        k (int): the number of nearest neighbours

    Returns:
        edge_feature (torch.Tensor): (batch_size, 2*num_dims, num_nodes, k)

    """
    pdist = pairwise_distance(features)
    _, knn_inds = torch.topk(pdist, k, largest=False)
    edge_feature = construct_edge_feature(features, knn_inds)

    return edge_feature
