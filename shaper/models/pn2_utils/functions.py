import torch

try:
    from shaper.models.pn2_utils import pn2_ext
except ImportError:
    print("Please compile source files before using pointnet2 cuda extension.")


def gather_points(points, index):
    """Gather xyz of centroids according to indices

    Args:
        points: (batch_size, channels, num_points)
        index: (batch_size, num_centroids)

    Returns:
        new_xyz (torch.Tensor): (batch_size, channels, num_centroids)

    """
    batch_size = points.size(0)
    channels = points.size(1)
    num_centroids = index.size(1)
    index_expand = index.unsqueeze(1).expand(batch_size, channels, num_centroids)
    return points.gather(2, index_expand)


class FarthestPointSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, num_centroids):
        """Farthest point sample

        Args:
            ctx:
            points (torch.Tensor): (batch_size, channels, num_points)
            num_centroids (int): the number of centroids to sample

        Returns:
            index (torch.Tensor): sample indices of centroids. (batch_size, num_centroids)

        """
        index = pn2_ext.farthest_point_sample(points, num_centroids)
        return index

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, None


farthest_point_sample = FarthestPointSample.apply


class BallQuery(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, centroids, radius, num_neighbours):
        """Ball query

        Args:
            ctx:
            points (torch.Tensor): (batch_size, channels, num_points)
            centroids (torch.Tensor): (batch_size, channels, num_centroids)
            radius (float): the radius of the ball
            num_neighbours (int): the number of neighbours within the ball.

        Returns:
            index (torch.Tensor): indices of neighbours of each centroid. (batch_size, num_centroids, num_neighbours)
            count (torch.Tensor): the number of unique neighbours of each centroid. (batch_size, num_centroids)

        """
        index, count = pn2_ext.ball_query(points, centroids, radius, num_neighbours)
        return index, count

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, None, None, None


ball_query = BallQuery.apply


class GroupPoints(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, index):
        """Group points by index

        Args:
            ctx:
            points (torch.Tensor): (batch_size, channels, num_points)
            index (torch.Tensor): indices of neighbours of each centroid. (batch_size, num_centroids, num_neighbours)

        Returns:
            group_points (torch.Tensor): grouped points. (batch_size, channels, num_centroids, num_neighbours)

        """
        ctx.save_for_backward(index)
        ctx.num_points = points.size(2)
        group_points = pn2_ext.group_points_forward(points, index)
        return group_points

    @staticmethod
    def backward(ctx, grad_output):
        index = ctx.saved_tensors[0]
        grad_input = pn2_ext.group_points_backward(grad_output, index, ctx.num_points)
        return grad_input, None


group_points = GroupPoints.apply
