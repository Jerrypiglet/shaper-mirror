from torch import nn

from .conv import Conv1d, Conv2d
from .linear import FC


class MLP(nn.Sequential):
    def __init__(self,
                 in_channels,
                 mlp_channels,
                 bn=True):
        """Multilayer perceptron

        Args:
            in_channels (int): the number of channels of input tensor
            mlp_channels (tuple): the numbers of channels of fully connected layers
            bn (bool): whether to use batch normalization
        """
        super(MLP, self).__init__()

        self.in_channels = in_channels

        for ind, out_channels in enumerate(mlp_channels):
            module = FC(in_channels, out_channels, relu=True, bn=bn)
            self.add_module(str(ind), module)
            in_channels = out_channels

        self.out_channels = in_channels


class SharedMLP(nn.Sequential):
    def __init__(self,
                 in_channels,
                 mlp_channels,
                 ndim=1,
                 bn=True):
        """Multilayer perceptron shared on resolution (1D or 2D)

        Args:
            in_channels (int): the number of channels of input tensor
            mlp_channels (tuple): the numbers of channels of fully connected layers
            ndim (int): the number of dimensions to share
            bn (bool): whether to use batch normalization
        """
        super(SharedMLP, self).__init__()

        self.in_channels = in_channels

        if ndim == 1:
            mlp_module = Conv1d
        elif ndim == 2:
            mlp_module = Conv2d
        else:
            raise ValueError()

        for ind, out_channels in enumerate(mlp_channels):
            module = mlp_module(in_channels, out_channels, 1, relu=True, bn=bn)
            self.add_module(str(ind), module)
            in_channels = out_channels

        self.out_channels = in_channels