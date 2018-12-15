"""
Module to project point clouds to sphere grid
"""

import numpy as np

import torch
import torch.nn as nn
import lie_learn.spaces.S2 as S2


def cal_sph_weight(xyz, grid, epsilon=1):
    """
    Function to calculate the projection weight of each grid
    Args:
        xyz: tensor (n_pc, 3)
        grid: tensor (2*band_width*2*band_width, 3)

    Returns:
        sph_weight: tensor (n_pc, 2*band_width*2*band_width)
    """
    n_pc = list(xyz.size())[0]
    n_grid = list(grid.size())[0]
    norm_pc = torch.norm(xyz, 2, dim=1, keepdim=True).repeat(1, n_grid)  # [n_pc, n_grid]
    norm_pc = torch.clamp(norm_pc, min=1e-6)
    vec_product = torch.mm(xyz, torch.transpose(grid, 0, 1))  # [n_pc, n_grid]
    quotient = torch.clamp(vec_product.div(norm_pc), min=-1.0, max=1.0)
    angle = torch.acos(quotient)
    sph_weight = torch.exp(-epsilon * angle)

    return sph_weight.type(torch.float32)


def get_projection_grid(b, grid_type="SOFT"):
    """
    returns the spherical grid in euclidean coordinates
    """
    theta, phi = S2.meshgrid(b=b, grid_type=grid_type)
    grid = S2.change_coordinates(np.c_[theta[..., None], phi[..., None]], p_from='S', p_to='C')
    grid = grid.reshape((-1, 3)).astype(np.float32)
    return grid


class PointCloudProjector(nn.Module):
    """
    Module to project point clouds to sphere grid for s2cnn
    Parameters:
        bandwidth: int
    Input:
        point_cloud: 4d tensor (batch_size, num_channels, num_points, num_samples) for grouped point cloud
                     or 3d tensor (batch_size, num_channels, num_points) for single point cloud
                     num_channels = 3 or 6 (XYZ or XYZ+normal)
        pts_cnt: 2d tensor, int, (batch_size, num_points)
                 number of unique points in each group

        
    Returns:
        sphere_grid: 5d tensor (batch_size, 2*bandwidth, 2*bandwidth, num_points, num_samples)
                     or 4d tensor (batch_size, 2*bandwidth, 2*bandwidth, num_points)
    """

    def __init__(self, bandwidth, grid_type="Driscoll-Healy"):
        super(PointCloudProjector, self).__init__()

        self.bandwidth = bandwidth
        self.grid = torch.as_tensor(get_projection_grid(self.bandwidth, grid_type)).type(torch.float32)
        self.gridT = torch.transpose(self.grid, 0, 1).type(torch.float32)

    def forward(self, x, pts_cnt=None):
        grid = self.grid.to(x.device)
        gridT = self.gridT.to(x.device)
        single_pc = False
        use_normal = False
        if len(list(x.size())) == 3:
            single_pc = True
            x.unsqueeze_(2)
            # print("input shape: ", list(x.size()))
        batch_size, num_channels, num_points, num_samples = list(x.size())
        # print('batch_size: ', batch_size)
        # print('num_channels: ', num_channels)
        # print('num_points: ', num_points)
        # print('num_samples: ', num_samples)
        assert (num_channels in [3, 6])
        xyz = x.narrow(1, 0, 3)
        if num_channels == 6:
            use_normal = True
            normal = x.narrow(1, 3, 3)

        if pts_cnt is None:
            pts_cnt = np.ones((batch_size, num_points)) * num_samples
            pts_cnt = torch.as_tensor(pts_cnt)
            pts_cnt = pts_cnt.to(x.device)

        # generate unique mask
        pts_cnt = pts_cnt.view(-1, 1).repeat(1, num_samples).type(torch.float32)  # [b*np, ns]
        nsrange = torch.arange(num_samples, dtype=torch.float32).unsqueeze(0).repeat(batch_size * num_points, 1).to(
            x.device)
        ones = torch.ones_like(nsrange, dtype=torch.float32).to(x.device)
        zeros = torch.zeros_like(nsrange, dtype=torch.float32).to(x.device)
        unique_mask = torch.where(nsrange < pts_cnt, ones, zeros).to(x.device)
        unique_mask_weighted = unique_mask.div(pts_cnt)  # [b*np, ns]

        # project xyz onto sphere grid
        # print('xyz shape at pos1: ', list(xyz.size()))
        xyz = torch.transpose(xyz, 1, 2).contiguous()
        # print('xyz shape at pos2: ', list(xyz.size()))
        xyz = xyz.view(batch_size * num_points, 3, num_samples)  # [b*np, 3, ns]
        # print('xyz shape at pos3: ', list(xyz.size()))
        xyz_flat = torch.transpose(xyz, 1, 2).contiguous().view(-1, 3)
        weight_sph = cal_sph_weight(xyz_flat, grid)  # [b*np*ns, ng]

        r_pc = torch.norm(xyz, 2, dim=1).type(torch.float32)  # [b*np, ns]
        r_pc = torch.mul(r_pc, unique_mask_weighted)
        r_pc_flat = r_pc.view(-1, 1).repeat(1, 2 * self.bandwidth * 2 * self.bandwidth)  # [b*np*ns, ng]

        grid_val_xyz = torch.mul(r_pc_flat, weight_sph).contiguous() \
            .view(batch_size * num_points, num_samples, 2 * self.bandwidth * 2 * self.bandwidth)

        grid_val_xyz = torch.transpose(grid_val_xyz, 1, 2)  # [b*np, ng, ns]
        grid_val_xyz = torch.sum(grid_val_xyz, 2)  # [b*np, ng]
        grid_val_xyz = grid_val_xyz.view(batch_size, num_points, -1)
        grid_val_xyz = torch.transpose(grid_val_xyz, 1, 2).contiguous() \
            .view(batch_size, 2 * self.bandwidth, 2 * self.bandwidth, num_points)

        if single_pc:
            grid_val_xyz = torch.squeeze(grid_val_xyz, dim=3)

        # project normals onto sphere grid
        if use_normal:
            normal = torch.transpose(normal, 1, 2).contiguous() \
                .view(-1, 3, num_samples).type(torch.float32)  # [b*np, 3, ns]
            normal = torch.mul(normal, unique_mask_weighted.unsqueeze(1).repeat(1, 3, 1))
            normal_flat = torch.transpose(normal, 1, 2).contiguous() \
                .view(-1, 3)  # [b*np*ns, 3]
            normal_grid_product = torch.mm(normal_flat, gridT)  # [b*np*ns, ng]

            grid_val_normal = torch.mul(normal_grid_product, weight_sph) \
                .view(batch_size * num_points, num_samples, 2 * self.bandwidth * 2 * self.bandwidth)

            grid_val_normal = torch.transpose(grid_val_normal, 1, 2)  # [b*np, ng, ns]
            grid_val_normal = torch.sum(grid_val_normal, 2)  # [b*np, ng]
            grid_val_normal = grid_val_normal.view(batch_size, num_points, -1)
            grid_val_normal = torch.transpose(grid_val_normal, 1, 2) \
                .view(batch_size, 2 * self.bandwidth, 2 * self.bandwidth, num_points)

            if single_pc:
                grid_val_normal = torch.squeeze(grid_val_normal, dim=3)

            return [grid_val_xyz.unsqueeze_(1), grid_val_normal.unsqueeze_(1)]

        else:
            return grid_val_xyz.unsqueeze_(1)


if __name__ == "__main__":
    grid = get_projection_grid(4)
    print(grid.shape)

    pc_projector = PointCloudProjector(bandwidth=16)
    print('grid: ', pc_projector.grid.shape)
    # print(pc_projector.grid)
    np.savetxt('/home/rayc/Projects/shaper/trials/grid.txt', pc_projector.grid, fmt="%.4f")
    pc = np.random.randn(2, 3, 2)
    pc = torch.as_tensor(pc).type(torch.float32)
    grid_val = pc_projector(pc)
    print('pc projection shape: ', grid_val.shape)

    np.savetxt('/home/rayc/Projects/shaper/trials/pc.txt', np.transpose(np.squeeze(pc[0, ...])), fmt="%.4f")
    grid_pc = np.concatenate((pc_projector.grid, np.reshape(grid_val[0, 0, ...], (-1, 1))), axis=-1)
    np.savetxt('/home/rayc/Projects/shaper/trials/pc_grid.txt', grid_pc, fmt="%.4f")

    pcn = np.random.rand(4, 6, 20)
    pcn = torch.as_tensor(pcn).type(torch.float32)
    grid_val_pcn = pc_projector(pcn)
    print('pcn projection shape: ', grid_val_pcn[0].shape, grid_val_pcn[1].shape)
