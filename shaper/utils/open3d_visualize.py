import os
import numpy as np
import open3d


# TODO: add more visualization methods
class Visualizer:
    map_label_to_rgb = {
        1: [0, 255, 0],
        2: [0, 0, 255],
        3: [255, 0, 0],
        4: [255, 0, 255],  # purple
        5: [0, 255, 255],  # cyan
        6: [255, 255, 0],  # yellow
    }

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def visualize(self, obj_category, obj_id):
        # Concat paths
        pts_path = os.path.join(self.dataset_path, obj_category,
                                'points', obj_id + '.pts')
        label_path = os.path.join(self.dataset_path, obj_category,
                                  'points_label', obj_id + '.seg')

        # Read point cloud
        point_cloud = open3d.read_point_cloud(pts_path, format='xyz')
        print(point_cloud)

        # Read label and map to color
        labels = np.loadtxt(label_path)
        colors = np.array(
            [Visualizer.map_label_to_rgb[label] for label in labels])
        point_cloud.colors = open3d.Vector3dVector(colors)
        open3d.draw_geometries([point_cloud])

    @staticmethod
    def visualize_pts_with_color(pts, color=np.array([[0, 0, 0]])):
        assert (color.shape[0] == 1 or color.shape[0] == pts.shape[0])
        # print('pts shape: ', pts.shape)
        # print('color shape: ', color.shape)
        if color.shape[0] == 1:
            color = np.tile(color, [pts.shape[0], 1])
        # print('color shape: ', color.shape)
        point_cloud = open3d.PointCloud()
        point_cloud.points = open3d.Vector3dVector(pts)
        point_cloud.colors = open3d.Vector3dVector(color)
        open3d.draw_geometries([point_cloud])

    @staticmethod
    def visualize_pts(pts):
        point_cloud = open3d.PointCloud()
        point_cloud.points = open3d.Vector3dVector(pts)
        open3d.draw_geometries([point_cloud])


if __name__ == '__main__':
    dataset_path = '../data/shapenet'
    v = Visualizer(dataset_path)
    v.visualize('02691156', '1a04e3eab45ca15dd86060f189eb133')
