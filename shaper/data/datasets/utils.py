import numpy as np


def crop_or_pad_points(points, num_points=-1, shuffle=False):
    """Crop or pad point cloud to a fixed number

    Args:
        points (np.ndarray): points cloud. (n, d)
        num_points (int): the number of output points
        shuffle (bool): whether to shuffle the order

    Returns:
        np.ndarray: output points cloud
        np.ndarray: index to choose input points

    """
    if shuffle:
        choice = np.random.permutation(len(points))
    else:
        choice = np.arange(len(points))
    if num_points > 0:
        if len(points) >= num_points:
            choice = choice[:num_points]
        else:
            num_pad = num_points - len(points)
            pad = np.random.choice(choice, num_pad, replace=True)
            choice = np.concatenate([choice, pad])
    points = points[choice]
    return points, choice


def normalize_points(points):
    """Normalize point cloud

    Args:
        points (np.ndarray): (n, 3)

    Returns:
        np.ndarray

    """
    assert points.ndim == 2 and points.shape[1] == 3
    centroid = np.mean(points, axis=0)
    points = points - centroid
    norm = np.max(np.linalg.norm(points, ord=2, axis=1))
    points = points / norm
    return points
