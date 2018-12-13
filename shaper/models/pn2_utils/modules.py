import torch
import torch.nn as nn

from shaper.nn import SharedMLP
from . import functions as _F


class FarthestPointSampler(nn.Module):
    """Farthest point sampling

    Args:
        num_centroids (int): the number of centroids

    """
    def __init__(self, num_centroids):
        super(FarthestPointSampler, self).__init__()
        self.num_centroids = num_centroids

    def forward(self, point):
        index = _F.farthest_point_sample(point, self.num_centroids)
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
            feature (torch.Tensor or None): (batch_size, in_channels, num_points)

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
            feature (torch.Tensor or None): (batch_size, in_channels, num_points)

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
