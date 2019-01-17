import torch
from shaper.nn.functional import pdist

try:
    from shaper.models.pn2_utils import pn2_ext
except ImportError:
    print("Please compile source files before using pointnet2 cuda extension.")


def gather_points(point, index):
    """Gather xyz of centroids according to indices

    Args:
        point: (batch_size, channels, num_points)
        index: (batch_size, num_centroids)

    Returns:
        new_xyz (torch.Tensor): (batch_size, channels, num_centroids)

    """
    batch_size = point.size(0)
    channels = point.size(1)
    num_centroids = index.size(1)
    index_expand = index.unsqueeze(1).expand(batch_size, channels, num_centroids)
    return point.gather(2, index_expand)


class FarthestPointSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, point, num_centroids):
        distance = pdist(point)
        mdist, pos = distance.max(2)[0].max(1)   
        index = pn2_ext.farthest_point_sample(mdist, pos, distance, point, num_centroids)
        return index

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, None


farthest_point_sample = FarthestPointSample.apply


class BallQuery(torch.autograd.Function):
    @staticmethod
    def forward(ctx, point, centroid, radius, num_neighbours):
        index, count = pn2_ext.ball_query(point, centroid, radius, num_neighbours)
        return index, count

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, None, None, None


ball_query = BallQuery.apply


class GroupPoints(torch.autograd.Function):
    @staticmethod
    def forward(ctx, point, index):
        ctx.save_for_backward(index)
        ctx.num_points = point.size(2)
        grouped_point = pn2_ext.group_points_forward(point, index)
        return grouped_point

    @staticmethod
    def backward(ctx, grad_output):
        index = ctx.saved_tensors[0]
        grad_input = pn2_ext.group_points_backward(grad_output, index, ctx.num_points)
        return grad_input, None


group_points = GroupPoints.apply
