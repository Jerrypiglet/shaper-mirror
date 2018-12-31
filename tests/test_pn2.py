import numpy as np
import torch
from torch.autograd import gradcheck

from shaper.models.pointnet2.functions import farthest_point_sample, group_points, ball_query
from shaper.models.pointnet2.functions import search_nn_distance, feature_interpolate


def farthest_point_sample_np(points, num_centroids):
    """Farthest point sample

    Args:
        points: (batch_size, 3, num_points)
        num_centroids (int): the number of centroids

    Returns:
        index (np.ndarray): index of centroids. (batch_size, num_centroids)

    """
    index = []
    for points_per_batch in points:
        index_per_batch = [0]
        cur_ind = 0
        dist2set = None
        for ind in range(1, num_centroids):
            cur_xyz = points_per_batch[:, cur_ind]
            dist2cur = points_per_batch - cur_xyz[:, None]
            dist2cur = np.square(dist2cur).sum(0)
            if dist2set is None:
                dist2set = dist2cur
            else:
                dist2set = np.minimum(dist2cur, dist2set)
            cur_ind = np.argmax(dist2set)
            index_per_batch.append(cur_ind)
        index.append(index_per_batch)
    return np.asarray(index)


def test_farthest_point_sample():
    batch_size = 16
    channels = 3
    num_points = 1024
    num_centroids = 128

    np.random.seed(0)
    points = np.random.rand(batch_size, channels, num_points)

    # points = []
    # for b in range(batch_size):
    #     x, y, z = np.meshgrid(np.arange(4), np.arange(4), np.arange(4))
    #     points.append(np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], 0))
    # points = np.asarray(points).astype(float)

    index = farthest_point_sample_np(points, num_centroids)
    point_tensor = torch.from_numpy(points).cuda()
    index_tensor = farthest_point_sample(point_tensor, num_centroids)
    index_tensor = index_tensor.cpu().numpy()
    assert np.all(index == index_tensor)


def test_group_points():
    torch.manual_seed(0)
    batch_size = 2
    num_inst = 512
    num_select = 128
    channels = 64
    k = 64

    feature = torch.rand(batch_size, channels, num_inst).cuda(0)
    index = torch.randint(0, num_inst, [batch_size, num_select, k]).long().cuda(0)

    feature_gather = torch.zeros_like(feature).copy_(feature)
    feature_gather.requires_grad = True
    feature_cuda = torch.zeros_like(feature).copy_(feature)
    feature_cuda.requires_grad = True

    feature_expand = feature_gather.unsqueeze(2).expand(batch_size, channels, num_select, num_inst)
    index_expand = index.unsqueeze(1).expand(batch_size, channels, num_select, k)
    out_gather = torch.gather(feature_expand, 3, index_expand)

    out_cuda = group_points(feature_cuda, index)
    assert out_gather.allclose(out_cuda)

    out_gather.backward(torch.ones_like(out_gather))
    out_cuda.backward(torch.ones_like(out_cuda))
    grad_gather = feature_gather.grad
    grad_cuda = feature_cuda.grad
    assert grad_gather.allclose(grad_cuda)


def ball_query_np(points, centroids, radius, num_neighbours):
    index = []
    count = []
    num_centroids = centroids.shape[2]

    for centroids_per_batch, points_per_batch in zip(centroids, points):
        index_per_batch = []
        count_per_batch = []
        for i in range(num_centroids):
            cur_centroid = centroids_per_batch[:, i]
            dist2cur = points_per_batch - cur_centroid[:, None]
            dist2cur = np.square(dist2cur).sum(0)
            neighbour_index = np.nonzero(dist2cur < (radius ** 2))[0]
            assert neighbour_index.size > 0
            count_per_batch.append(min(neighbour_index.size, num_neighbours))

            if neighbour_index.size < num_neighbours:
                neighbour_index = np.concatenate([neighbour_index,
                                                  np.repeat(neighbour_index[0], num_neighbours - neighbour_index.size)])
            else:
                neighbour_index = neighbour_index[:num_neighbours]

            index_per_batch.append(neighbour_index)
        index.append(index_per_batch)
        count.append(count_per_batch)
    return np.asarray(index), np.asarray(count)


def test_ball_query():
    batch_size = 2
    channels = 3
    num_points = 8
    num_centroids = 3
    radius = 0.2
    num_neighbours = 4

    np.random.seed(0)
    points = np.random.rand(batch_size, channels, num_points)
    centroids = np.asarray([p[:, np.random.choice(num_points, [num_centroids], replace=False)] for p in points])
    index, count = ball_query_np(points, centroids, radius, num_neighbours)

    points_tensor = torch.from_numpy(points).cuda()
    centroids_tensor = torch.from_numpy(centroids).cuda()
    index_tensor, count_tensor = ball_query(points_tensor, centroids_tensor, radius, num_neighbours)
    index_tensor = index_tensor.cpu().numpy()
    count_tensor = count_tensor.cpu().numpy()

    assert np.all(index == index_tensor)
    assert np.all(count == count_tensor)


def search_nn_distance_np(query_xyz, key_xyz, num_neighbors):
    """For each point in query set, find its distances to k nearest neighbors in key set

    Args:
        query_xyz: (batch_size, 3, num_query)
        key_xyz: (batch_size, 3, num_key)
        num_neighbors (int):

    Returns:
        distance: (batch_size, num_query, num_neighbor), distance to the k nearest neighbors in key_xyz
        index: (batch_size, num_query, num_neighbor), indices of these neighbors in key_xyz
    """
    num_query = query_xyz.shape[2]
    num_key = key_xyz.shape[2]
    assert num_neighbors <= num_query and num_neighbors <= num_key
    assert query_xyz.shape[0] == key_xyz.shape[0]
    assert query_xyz.shape[1] == key_xyz.shape[1]

    distance = []
    index = []

    for query_xyz_per_batch, key_xyz_per_batch in zip(query_xyz, key_xyz):
        distance_per_batch = []
        index_per_batch = []
        for cur_ind in range(num_query):
            cur_xyz = query_xyz_per_batch[:, cur_ind]
            diff = key_xyz_per_batch - cur_xyz[:, None]
            dist2key = np.sum(diff ** 2, axis=0)
            sorted_idx = np.argsort(dist2key)
            idx_knn = sorted_idx[:num_neighbors]
            dist_knn = dist2key[idx_knn]
            index_per_batch.append(idx_knn)
            distance_per_batch.append(dist_knn)
        distance.append(distance_per_batch)
        index.append(index_per_batch)

    distance = np.asarray(distance)
    index = np.asarray(index)
    return distance, index


def test_search_nn_distance():
    batch_size = 4
    channels = 3
    num_query = 512
    num_key = 128
    num_neighbors = 3

    np.random.seed(0)
    query_xyz = np.random.rand(batch_size, channels, num_query)
    key_xyz = np.random.rand(batch_size, channels, num_key)
    distance, index = search_nn_distance_np(query_xyz, key_xyz, num_neighbors)

    query_xyz_tensor = torch.from_numpy(query_xyz).cuda()
    key_xyz_tensor = torch.from_numpy(key_xyz).cuda()
    distance_tensor, index_tensor = search_nn_distance(query_xyz_tensor, key_xyz_tensor, num_neighbors)
    distance_tensor = distance_tensor.cpu().numpy()
    index_tensor = index_tensor.cpu().numpy()

    np.testing.assert_allclose(distance, distance_tensor, verbose=True)
    np.testing.assert_equal(index, index_tensor, verbose=True)


def feature_interpolate_np(feature, index, weight):
    """
    Generate new features based on input features

    Args:
        feature: (batch_size, channels, num_key)
        index: (batch_size, num_query, num_neighbours)
        weight: (batch_size, num_query, num_neighbours)

    Returns:
        interpolated feature: (batch_size, channels, num_query)
    """
    batch_size, _, num_key = feature.shape
    num_query = index.shape[1]
    assert batch_size == index.shape[0]
    assert all(x == y for x, y in zip(index.shape, weight.shape))

    interpolated_feature = []
    for batch_ind in range(batch_size):
        feature_per_batch = feature[batch_ind]
        index_per_batch = index[batch_ind]
        weight_per_batch = weight[batch_ind]

        interpolated_feature_per_batch = []
        for query_ind in range(num_query):
            cur_index = index_per_batch[query_ind]
            cur_weight = weight_per_batch[query_ind]
            cur_feature = feature_per_batch[:, cur_index]
            new_feature = np.matmul(cur_feature, cur_weight)
            interpolated_feature_per_batch.append(new_feature)
        interpolated_feature.append(interpolated_feature_per_batch)

    interpolated_feature = np.asarray(interpolated_feature)
    interpolated_feature = interpolated_feature.swapaxes(1, 2)

    return interpolated_feature


def test_feature_interpolate():
    batch_size = 4
    channels = 3
    num_query = 64
    num_key = 32
    num_neighbors = 3

    feature = np.random.rand(batch_size, channels, num_key)
    index = np.random.randint(num_key, size=(batch_size, num_query, num_neighbors))
    weight = np.random.uniform(1e-10, 1, [batch_size, num_query, num_neighbors])
    weight = weight / weight.sum(axis=2, keepdims=True)

    new_feature = feature_interpolate_np(feature, index, weight)

    features_tensor = torch.from_numpy(feature).cuda()
    index_tensor = torch.from_numpy(index).cuda()
    weight_tensor = torch.from_numpy(weight).cuda()
    new_feature_tensor = feature_interpolate(features_tensor, index_tensor, weight_tensor)
    new_feature_tensor = new_feature_tensor.cpu().numpy()

    np.testing.assert_allclose(new_feature, new_feature_tensor)

    features_tensor.requires_grad = True
    assert gradcheck(feature_interpolate, (features_tensor, index_tensor, weight_tensor))
