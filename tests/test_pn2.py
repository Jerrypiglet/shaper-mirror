import numpy as np
import torch

from shaper.models.pointnet2.functions import farthest_point_sample, group_points, ball_query


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


def test_point_search():
    # this is a draft for testing the point_search cuda code
    pass
    # dist, idx = _F.search_nn_distance(xyz1, xyz2, self.num_neighbors)
    # dist = torch.clamp(dist, min=1e-10)
    #
    # print("dist:\n", dist.size())
    # print("xyz2:\n", xyz2.size())
    # print("xyz1:\n", xyz1.size())
    #
    # dist_repl = 1.0 / dist
    # print("dist_repl:\n", dist_repl.size())
    # print(dist_repl[0])
    #
    # norm = torch.sum(1.0 / dist, dim=1, keepdim=True)
    # print("norm:\n", norm.size())
    # print(norm[0])
    # exit()
    # weight = (1.0 / dist) / norm
