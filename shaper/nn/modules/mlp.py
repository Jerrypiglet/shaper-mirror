from torch import nn
import torch.nn.functional as F

from .conv import Conv1d, Conv2d
from .linear import FC


class MLP(nn.ModuleList):
    """Multilayer perceptron

    Args:
        in_channels (int): the number of channels of input tensor
        mlp_channels (tuple): the numbers of channels of fully connected layers
        dropout (float, None or tuple of length of mlp_channels of float or None): dropout ratio
        bn (bool): whether to use batch normalization

    """

    def __init__(self,
                 in_channels,
                 mlp_channels,
                 dropout=None,
                 bn=True,
                 bn_momentum=0.1):
        super(MLP, self).__init__()

        self.in_channels = in_channels
        # define dropout tuple (must have same length as mlp_channels)
        if dropout:
            if type(dropout) == tuple:
                assert len(dropout) == len(mlp_channels)
                self.dropout = dropout
            else:
                assert type(dropout) == float
                self.dropout = (dropout,) * len(mlp_channels)
        else:
            self.dropout = (None,) * len(mlp_channels)

        for ind, out_channels in enumerate(mlp_channels):
            self.append(FC(in_channels, out_channels,
                           relu=True, bn=bn, bn_momentum=bn_momentum))
            in_channels = out_channels

        self.out_channels = in_channels

    def forward(self, x):
        end_points = {}
        for i, module in enumerate(self):
            x = module(x)
            if self.dropout[i]:
                # FIXME: Pytorch 1.0 Raise Error
                x = F.dropout(x, self.dropout[i], self.training, inplace=False)
            end_points["out{}".format(i + 1)] = x
        return x, end_points


class SharedMLP(nn.ModuleList):
    def __init__(self,
                 in_channels,
                 mlp_channels,
                 ndim=1,
                 relu=True,
                 dropout=None,
                 bn=True,
                 bn_momentum=0.1):
        """Multilayer perceptron shared on resolution (1D or 2D)

        Args:
            in_channels (int): the number of channels of input tensor
            mlp_channels (tuple): the numbers of channels of fully connected layers
            ndim (int): the number of dimensions to share
            bn (bool): whether to use batch normalization
        """
        super(SharedMLP, self).__init__()

        self.in_channels = in_channels
        # define dropout tuple (must have same length as mlp_channels)
        if dropout:
            if type(dropout) == tuple:
                assert len(dropout) == len(mlp_channels)
                self.dropout = dropout
            else:
                assert type(dropout) == float
                self.dropout = (dropout,) * len(mlp_channels)
        else:
            self.dropout = (None,) * len(mlp_channels)

        if ndim == 1:
            mlp_module = Conv1d
        elif ndim == 2:
            mlp_module = Conv2d
        else:
            raise ValueError()

        for ind, out_channels in enumerate(mlp_channels):
            self.append(mlp_module(in_channels, out_channels, 1,
                                   relu=relu, bn=bn, bn_momentum=bn_momentum))
            in_channels = out_channels

        self.out_channels = in_channels

    def forward(self, x):
        end_points = {}
        for i, module in enumerate(self):
            x = module(x)
            if self.dropout[i]:
                # FIXME: Pytorch 1.0 Raise Error
                x = F.dropout(x, self.dropout[i], self.training, inplace=False)
            end_points["out{}".format(i + 1)] = x
        return x, end_points
