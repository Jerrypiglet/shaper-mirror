import torch

try:
    from shaper.models.pointnet2 import pn2_ext
except ImportError:
    print("Please compile source files before using pointnet2 cuda extension.")


def gather_points(points, index):
    """Gather xyz of centroids according to indices

    Args:
        points: (batch_size, channels, num_points)
        index: (batch_size, num_centroids)

    Returns:
        new_xyz (torch.Tensor): (batch_size, channels, num_centroids)

    """
    batch_size = points.size(0)
    channels = points.size(1)
    num_centroids = index.size(1)
    index_expand = index.unsqueeze(1).expand(batch_size, channels, num_centroids)
    return points.gather(2, index_expand)


class FarthestPointSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, num_centroids):
        """Farthest point sample

        Args:
            ctx:
            points (torch.Tensor): (batch_size, channels, num_points)
            num_centroids (int): the number of centroids to sample

        Returns:
            index (torch.Tensor): sample indices of centroids. (batch_size, num_centroids)

        """
        index = pn2_ext.farthest_point_sample(points, num_centroids)
        return index

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, None


farthest_point_sample = FarthestPointSample.apply


class BallQuery(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, centroids, radius, num_neighbours):
        """Ball query

        Args:
            ctx:
            points (torch.Tensor): (batch_size, channels, num_points)
            centroids (torch.Tensor): (batch_size, channels, num_centroids)
            radius (float): the radius of the ball
            num_neighbours (int): the number of neighbours within the ball.

        Returns:
            index (torch.Tensor): indices of neighbours of each centroid. (batch_size, num_centroids, num_neighbours)
            count (torch.Tensor): the number of unique neighbours of each centroid. (batch_size, num_centroids)

        """
        index, count = pn2_ext.ball_query(points, centroids, radius, num_neighbours)
        return index, count

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, None, None, None


ball_query = BallQuery.apply


class GroupPoints(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, index):
        """Group points by index

        Args:
            ctx:
            points (torch.Tensor): (batch_size, channels, num_points)
            index (torch.Tensor): indices of neighbours of each centroid. (batch_size, num_centroids, num_neighbours)

        Returns:
            group_points (torch.Tensor): grouped points. (batch_size, channels, num_centroids, num_neighbours)

        """
        ctx.save_for_backward(index)
        ctx.num_points = points.size(2)
        group_points = pn2_ext.group_points_forward(points, index)
        return group_points

    @staticmethod
    def backward(ctx, grad_output):
        index = ctx.saved_tensors[0]
        grad_input = pn2_ext.group_points_backward(grad_output, index, ctx.num_points)
        return grad_input, None


group_points = GroupPoints.apply


class SearchNNDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, num_neighbor):
        """For each element in xyz1, find its distances to k nearest neighbor in xyz2

        :param ctx:
        :param xyz1: (b, n, 3) xyz of the input of set abstraction layer
        :param xyz2: (b, m, 3) xyz of the output of set abstraction layer
        :param num_neighbor: k nearest neighbor
        :return:
            dist: (b, n, k) distance to the k nearest neighbors in xyz2
            idx: (b, n, k) indices of these neighbors in xyz2
        """
        # n = xyz1.size(1)
        # m = xyz2.size(1)
        dist, idx = pn2_ext.point_search(num_neighbor, xyz2, xyz1)
        return dist, idx

    @staticmethod
    def backward(ctx):
        return None, None


search_nn_distance = SearchNNDistance.apply


class FeatureInterpolation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        """

        :param ctx:
        :param features: (b, m, c) features of the input of set abstraction layer
        :param idx: (b, n, k) indices to the input
        :param weight: (b, n, k) weights to the input
        :return:
            interpolated_features: (b, n, c)
        """
        _, _, m = features.size()
        # _, n, k = idx.size()
        ctx.params_for_backward = (m, idx, weight)      # Save parameters for backward
        interpolated_features = pn2_ext.interpolate(features, idx, weight)
        return interpolated_features

    @staticmethod
    def backward(ctx, grad_out):
        """

        :param ctx:
        :param grad_out: (b, n, c) gradient outputs
        :return: (b, m, c)
        """
        #_, n, c = grad_out.size()
        m, idx, weight = ctx.params_for_backward

        ret_grad = pn2_ext.interpolate_backward(m, grad_out, weight, idx)
        return ret_grad


feature_interpolation = FeatureInterpolation.apply
