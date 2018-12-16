"""Open3D visualization tools

References:
    @article{Zhou2018,
        author    = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
        title     = {{Open3D}: {A} Modern Library for {3D} Data Processing},
        journal   = {arXiv:1801.09847},
        year      = {2018},
    }

See Also:
    https://github.com/IntelVCL/Open3D
    https://github.com/IntelVCL/Open3D-PointNet/blob/master/open3d_visualilze.py

"""

import numpy as np
import matplotlib.pyplot as plt
import open3d


class Visualizer(object):
    @staticmethod
    def visualize_points(points):
        point_cloud = open3d.PointCloud()
        point_cloud.points = open3d.Vector3dVector(points)
        open3d.draw_geometries([point_cloud])

    @staticmethod
    def visualize_points_with_labels(points, labels, cmap=None, lut=6):
        if cmap is None:
            cmap = plt.get_cmap("hsv", lut)
            cmap = np.asarray([cmap(i) for i in range(lut)])[:, :3]
        point_cloud = open3d.PointCloud()
        point_cloud.points = open3d.Vector3dVector(points)
        assert len(labels) == len(points)
        point_cloud.colors = open3d.Vector3dVector(cmap[labels])
        open3d.draw_geometries([point_cloud])


if __name__ == '__main__':
    dataset_path = '../data/shapenet'
    v = Visualizer(dataset_path)
    v.visualize('02691156', '1a04e3eab45ca15dd86060f189eb133')
