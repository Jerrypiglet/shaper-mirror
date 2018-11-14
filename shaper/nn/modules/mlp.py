from torch import nn

from .conv import Conv1d
from .linear import FC


class MLP(nn.Sequential):
    def __init__(self, in_channels,
                 mlp_spec=(512, 256),
                 bn=True):
        super(MLP, self).__init__()

        self.in_channels = in_channels

        for ind, out_channels in enumerate(mlp_spec):
            module = FC(in_channels, out_channels, relu=True, bn=bn)
            self.add_module(str(ind), module)
            in_channels = out_channels

        self.out_channels = in_channels


class SharedMLP(nn.Sequential):
    def __init__(self, in_channels,
                 mlp_spec=(64, 128, 1024),
                 bn=True):
        super(SharedMLP, self).__init__()

        self.in_channels = in_channels

        for ind, out_channels in enumerate(mlp_spec):
            module = Conv1d(in_channels, out_channels, 1, relu=True, bn=bn)
            self.add_module(str(ind), module)
            in_channels = out_channels

        self.out_channels = in_channels
