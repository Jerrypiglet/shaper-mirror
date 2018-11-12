import torch

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
