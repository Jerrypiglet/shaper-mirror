"""Helpers to transform point clouds. Especially for data augmentation

FIXME: Multi-thread loading will cause wrong behaviour of numpy random

"""

import torch
import numpy as np


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


def get_rot_mat(angle, axis):
    r"""Returns a 3x3 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)

    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                               [u[2], 0.0, -u[0]],
                               [-u[1], u[0], 0.0]])
    R = cos_angle * np.eye(3) + sin_angle * cross_prod_mat + (1.0 - cos_angle) * np.outer(u, u)
    R = torch.from_numpy(R).float()
    return R


class PointCloudRotate(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
        self.axis = axis

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = get_rot_mat(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return points @ rotation_matrix.t_()
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            rotation_matrix_t = rotation_matrix.t_()
            points[:, 0:3] = pc_xyz @ rotation_matrix_t
            points[:, 3:] = pc_normals @ rotation_matrix_t
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
        self.rotation_matrix = get_rot_mat(self.angle, self.axis)

    def __call__(self, points):
        normals = points.size(1) > 3
        if not normals:
            return points @ self.rotation_matrix.t()
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = pc_xyz @ self.rotation_matrix.t()
            points[:, 3:] = pc_normals @ self.rotation_matrix.t()


class PointCloudRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip

    def _get_angles(self):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip,
            self.angle_clip
        )

        return angles

    def __call__(self, points):
        angles = self._get_angles()
        Rx = get_rot_mat(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = get_rot_mat(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = get_rot_mat(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = Rz @ Ry @ Rx

        normals = points.size(1) > 3
        if not normals:
            return points @ rotation_matrix.t_()
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            rotation_matrix_t = rotation_matrix.t_()
            points[:, 0:3] = pc_xyz @ rotation_matrix_t
            points[:, 3:] = pc_normals @ rotation_matrix_t
            return points


class PointCloudTranslate(object):
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, points):
        translation = np.random.uniform(
            -self.translate_range, self.translate_range, [3])
        translation = torch.from_numpy(translation).float().to(points.device)
        translation = translation.unsqueeze(0).expand(points.size(0), 3)
        points[:, 0:3] += translation
        return points


class PointCloudScale(object):
    def __init__(self, lo=0.8, hi=1.25):
        self.lo, self.hi = lo, hi

    def __call__(self, points):
        scale = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scale
        return points


class PointCloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = points.new(points.size(0), 3).normal_(
            mean=0.0, std=self.std
        ).clamp_(-self.clip, self.clip)
        points[:, 0:3] += jittered_data
        return points


class PointCloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875):
        assert 0 <= max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points):
        dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
        dropout_indices = torch.nonzero(torch.rand(points.size(0)) <= dropout_ratio)[0]
        if dropout_indices.numel() > 0:
            points[dropout_indices] = points[0]  # set to the first point

        return points
