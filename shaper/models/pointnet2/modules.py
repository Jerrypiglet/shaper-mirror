import torch
import torch.nn as nn

from shaper.nn import SharedMLP
from . import functions as _F
from shaper.nn.functional import pdist

class FarthestPointSampler(nn.Module):
    """Farthest point sampling

    Args:
        num_centroids (int): the number of centroids

    """

    def __init__(self, num_centroids):
        super(FarthestPointSampler, self).__init__()
        self.num_centroids = num_centroids

    def forward(self, points):
        with torch.no_grad():
            index = _F.farthest_point_sample(points, self.num_centroids)
        return index

    def extra_repr(self):
        return "num_centroids={:d}".format(self.num_centroids)


class QueryGrouper(nn.Module):
    def __init__(self, radius, num_neighbours, use_xyz):
        super(QueryGrouper, self).__init__()
        self.radius = radius
        self.num_neighbours = num_neighbours
        self.use_xyz = use_xyz

    def forward(self, new_xyz, xyz, feature):
        with torch.no_grad():
            index, unique_count = _F.ball_query(xyz, new_xyz, self.radius, self.num_neighbours)

        # (batch_size, 3, num_centroids, num_neighbours)
        grouped_xyz = _F.group_points(xyz, index)
        # translation normalization
        grouped_xyz -= new_xyz.unsqueeze(-1)

        if feature is not None:
            # (batch_size, channels, num_centroids, num_neighbours)
            group_feature = _F.group_points(feature, index)
            if self.use_xyz:
                new_feature = torch.cat([grouped_xyz, group_feature], dim=1)
            else:
                new_feature = group_feature
        else:
            new_feature = grouped_xyz

        return new_feature

    def extra_repr(self):
        return "radius={:.1e}, num_neighbours={:d}, use_xyz={}".format(self.radius, self.num_neighbours, self.use_xyz)


class FeatureInterpolator(nn.Module):
    def __init__(self, num_neighbors, eps=1e-10):
        super(FeatureInterpolator, self).__init__()
        self.num_neighbors = num_neighbors
        self.eps = eps

    def forward(self, query_xyz, key_xyz, query_feature, key_feature):
        """

        Args:
            query_xyz: query xyz, (B, 3, N1)
            key_xyz: key xyz, (B, 3, N2)
            query_feature: (B, C1, N1), feature corresponding to xyz1
            key_feature: (B, C2, N2), feature corresponding to xyz2

        Returns:
            new_feature: (B, C1+C2, N2), propagated feature

        """
        with torch.no_grad():
            # distance: (B, N1, K)
            distance, index = _F.search_nn_distance(query_xyz, key_xyz, self.num_neighbors)
            distance = torch.clamp(distance, min=self.eps)
            inv_distance = 1.0 / distance
            norm = torch.sum(inv_distance, dim=2, keepdim=True)
            weight = inv_distance / norm

        interpolated_feature = _F.feature_interpolate(key_feature, index, weight)

        if query_feature is not None:
            new_feature = torch.cat([interpolated_feature, query_feature], dim=1)
        else:
            new_feature = interpolated_feature

        return new_feature

    def extra_repr(self):
        return "num_neighbours={:d}".format(self.num_neighbors)


class PointNetSAModule(nn.Module):
    """PointNet set abstraction module"""

    def __init__(self,
                 in_channels,
                 mlp_channels,
                 num_centroids,
                 radius,
                 num_neighbours,
                 use_xyz):
        super(PointNetSAModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = mlp_channels[-1]

        if use_xyz:
            in_channels += 3
        self.mlp = SharedMLP(in_channels, mlp_channels, ndim=2, bn=True)

        self.sampler = FarthestPointSampler(num_centroids)
        self.grouper = QueryGrouper(radius, num_neighbours, use_xyz=use_xyz)

    def forward(self, xyz, feature=None):
        """

        Args:
            xyz (torch.Tensor): (batch_size, 3, num_points)
                xyz coordinates of feature
            feature (torch.Tensor, optional): (batch_size, in_channels, num_points)

        Returns:
            new_xyz (torch.Tensor): (batch_size, 3, num_centroids)
            new_feature (torch.Tensor): (batch_size, out_channels, num_centroids)

        """
        # sample new points
        index = self.sampler(xyz)
        # (batch_size, 3, num_centroids)
        new_xyz = _F.gather_points(xyz, index)

        # (batch_size, in_channels, num_centroids, num_neighbours)
        new_feature = self.grouper(new_xyz, xyz, feature)

        new_feature = self.mlp(new_feature)
        new_feature, _ = torch.max(new_feature, 3)

        return new_xyz, new_feature

    def init_weights(self, init_fn=None):
        self.mlp.init_weights(init_fn)


class PointNetSAModuleMSG(nn.Module):
    """PointNet set abstraction module (multi scale)"""

    def __init__(self,
                 in_channels,
                 mlp_channels_list,
                 num_centroids,
                 radius_list,
                 num_neighbours_list,
                 use_xyz):
        super(PointNetSAModuleMSG, self).__init__()

        self.in_channels = in_channels
        self.out_channels = sum(mlp_channels[-1] for mlp_channels in mlp_channels_list)

        num_scales = len(mlp_channels_list)
        assert len(radius_list) == num_scales
        assert len(num_neighbours_list) == num_scales

        if use_xyz:
            in_channels += 3
        self.mlp = nn.ModuleList()

        self.sampler = FarthestPointSampler(num_centroids)
        self.grouper = nn.ModuleList()

        for ind in range(num_scales):
            self.mlp.append(SharedMLP(in_channels, mlp_channels_list[ind], ndim=2, bn=True))
            self.grouper.append(QueryGrouper(radius_list[ind], num_neighbours_list[ind], use_xyz=use_xyz))

    def forward(self, xyz, feature=None):
        """

        Args:
            xyz (torch.Tensor): (batch_size, 3, num_points)
                xyz coordinates of feature
            feature (torch.Tensor, optional): (batch_size, in_channels, num_points)

        Returns:
            new_xyz (torch.Tensor): (batch_size, 3, num_centroids)
            new_feature (torch.Tensor): (batch_size, out_channels, num_centroids)

        """
        # sample new points
        index = self.sampler(xyz)
        # (batch_size, 3, num_centroids)
        new_xyz = _F.gather_points(xyz, index)

        # multi-scale
        new_feature_list = []
        for mlp, grouper in zip(self.mlp, self.grouper):
            # (batch_size, in_channels, num_centroids, num_neighbours)
            new_feature = grouper(new_xyz, xyz, feature)
            new_feature = mlp(new_feature)
            new_feature, _ = torch.max(new_feature, 3)
            new_feature_list.append(new_feature)

        return new_xyz, torch.cat(new_feature_list, dim=1)

    def init_weights(self, init_fn=None):
        for mlp in self.mlp:
            mlp.init_weights(init_fn)


class PointnetFPModule(nn.Module):
    """PointNet feature propagation module"""

    def __init__(self,
                 in_channels,
                 mlp_channels,
                 num_neighbors):
        super(PointnetFPModule, self).__init__()

        self.in_channels = in_channels
        self.mlp = SharedMLP(in_channels, mlp_channels, ndim=1, bn=True)
        self.interpolator = FeatureInterpolator(num_neighbors)

    def forward(self, query_xyz, key_xyz, query_feature, key_feature):
        new_feature = self.interpolator(query_xyz, key_xyz, query_feature, key_feature)
        new_feature = self.mlp(new_feature)

        return new_feature

    def init_weights(self, init_fn=None):
        self.mlp.init_weights(init_fn)
