import numpy as np
import torch


def crop_or_pad_points(points, num_points=-1, shuffle=False):
    """Crop or pad point cloud to a fixed number

    Args:
        points (np.ndarray): point cloud. (n, d)
        num_points (int): the number of output points
        shuffle (bool): whether to shuffle the order

    Returns:
        np.ndarray: output point cloud
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

    # Pad with replacement (used in original PointNet++)
    # choice = np.random.choice(len(points), num_points, replace=True)

    # Return a copy to avoid operating original data
    points = points[choice].copy()

    return points, choice


def normalize_points(points):
    """Normalize point cloud

    Args:
        points (np.ndarray): (n, 3)

    Returns:
        np.ndarray: normalized points

    """
    assert points.ndim == 2 and points.shape[1] == 3
    centroid = np.mean(points, axis=0)
    points = points - centroid
    norm = np.max(np.linalg.norm(points, ord=2, axis=1))
    points = points / norm
    return points




def normalize_batch_points(points):
    """Normalize point cloud

    Args:
        points : (b, n, 3)

    Returns:
        np.ndarray: normalized points

    """
    assert  points.shape[2] == 3
    batch_size = points.shape[0]
    ##center zoomed points
    points -= torch.sum(points, 1,keepdim=True)/points.shape[1]

    maxnorm, _ = torch.max(torch.sum(points**2, 2),1)
    maxnorm = maxnorm ** 0.5
    maxnorm = maxnorm.view(batch_size, 1, 1)
    points /=maxnorm

    return points
