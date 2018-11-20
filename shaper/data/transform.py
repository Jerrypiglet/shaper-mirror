"""Helpers to transform point clouds. Especially for data augmentation"""

import numpy as np
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points):
        for t in self.transforms:
            points = t(points)
        return points

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class PointCloudToTensor(object):
    def __call__(self, points):
        assert isinstance(points, np.ndarray)
        return torch.as_tensor(points).float()


class PointCloudTensorTranspose(object):
    def __call__(self, points):
        return points.transpose_(0, 1)


def get_rotation_matrix_np(angle, axis):
    """Returns a 3x3 rotation matrix that performs a rotation around axis by angle
    Numpy version

    Args:
        angle (float or torch.Tensor): Angle to rotate by
        axis (np.ndarray): Axis to rotate about

    Returns:
        np.ndarray: 3x3 rotation matrix A. (y=A'x)

    """
    u = axis / np.linalg.norm(axis)
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    cross_prod_mat = np.cross(u, np.eye(3))
    R = cos_angle * np.eye(3) + sin_angle * cross_prod_mat + (1.0 - cos_angle) * np.outer(u, u)
    return R


def get_rotation_matrix(angle, axis):
    """Return a rotation matrix by an angle around a given axis
    Pytorch version is slightly slower than numpy version.

    Args:
        angle (float):
        axis (torch.Tensor): (3,)

    Returns:
        R (torch.Tensor): rotation matrix (3, 3) A. (y=A'x)

    """
    assert axis.numel() == 3
    u = axis / torch.norm(axis)
    u = u.view(1, -1)
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    cross_product_matrix = torch.cross(u.expand(3, 3), torch.eye(3), dim=1)
    # Not necessary to transpose here
    R = cos_angle * torch.eye(3) + sin_angle * cross_product_matrix + (1 - cos_angle) * (u.t() @ u)
    return R


class PointCloudRotate(object):
    def __init__(self, axis=(0.0, 1.0, 0.0)):
        self.axis = torch.as_tensor(axis).float()

    def __call__(self, points):
        angle = torch.rand(1) * 2 * np.pi
        rotation_matrix = get_rotation_matrix(angle, self.axis)

        use_normal = points.size(1) > 3
        if not use_normal:
            return points @ rotation_matrix
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = pc_xyz @ rotation_matrix
            points[:, 3:] = pc_normals @ rotation_matrix
            return points


class PointCloudRotateByAngle(object):
    def __init__(self, axis_name, angle):
        assert axis_name in ["x", "y", "z"]
        if axis_name == "x":
            self.axis = np.array([1.0, 0.0, 0.0])
        elif axis_name == "y":
            self.axis = np.array([0.0, 1.0, 0.0])
        else:
            self.axis = np.array([0.0, 0.0, 1.0])
        self.angle = angle
        rotation_matrix = get_rotation_matrix_np(self.angle, self.axis)
        self.rotation_matrix = torch.from_numpy(rotation_matrix).float()

    def __call__(self, points):
        use_normal = points.size(1) > 3
        if not use_normal:
            return points @ self.rotation_matrix
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = pc_xyz @ self.rotation_matrix
            points[:, 3:] = pc_normals @ self.rotation_matrix
            return points


class PointCloudRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip
        self.axes = torch.eye(3)

    def __call__(self, points):
        angles = torch.clamp(self.angle_sigma * torch.randn(3),
                             -self.angle_clip, self.angle_clip)

        Rx = get_rotation_matrix(angles[0], self.axes[0])
        Ry = get_rotation_matrix(angles[1], self.axes[1])
        Rz = get_rotation_matrix(angles[2], self.axes[2])

        rotation_matrix = Rz @ Ry @ Rx

        use_normal = points.size(1) > 3
        if not use_normal:
            return points @ rotation_matrix
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = pc_xyz @ rotation_matrix
            points[:, 3:] = pc_normals @ rotation_matrix
            return points


class PointCloudTranslate(object):
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, points):
        translation = points.new(3).uniform_(-self.translate_range, self.translate_range)
        points[:, 0:3] += translation  # broadcast first dimension
        return points


class PointCloudScale(object):
    def __init__(self, lo=0.8, hi=1.25):
        assert hi >= lo
        self.lo = lo
        self.hi = hi

    def __call__(self, points):
        scale = points.new(1).uniform_(self.lo, self.hi)
        points[:, 0:3] *= scale
        return points


class PointCloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = points.new(points.size(0), 3).normal_(mean=0.0, std=self.std)
        jittered_data = jittered_data.clamp_(-self.clip, self.clip)
        points[:, 0:3] += jittered_data
        return points


class PointCloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875):
        assert 0 <= max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points):
        dropout_ratio = torch.rand() * self.max_dropout_ratio
        dropout_indices = torch.nonzero(torch.rand(points.size(0)) <= dropout_ratio)[0]
        if dropout_indices.numel() > 0:
            points[dropout_indices] = points[0]  # set to the first point
        return points


def test_rotation_matrix():
    axis = np.array([-0.5, 1., 0.5]).astype(np.float32)
    angle = 1.0
    rotation_matrix_np = get_rotation_matrix_np(angle, axis).astype(np.float32)
    rotation_matrix_tensor = get_rotation_matrix(
        angle=torch.tensor(angle).float(),
        axis=torch.tensor(axis))
    print(np.allclose(rotation_matrix_np, rotation_matrix_tensor))


if __name__ == '__main__':
    test_rotation_matrix()
