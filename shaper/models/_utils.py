import torch
from torch import nn
import torch.nn.functional as F
from typing import List

class SharedMLP(nn.Sequential):

    def __init__(
            self,
            args: List[int],
            *,
            bn: bool = False,
            activation=nn.ReLU(inplace=True),
            preact: bool = False,
            first: bool = False,
            name: str = ""
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    kernel_size=1,
                    bn=(not first or (i != 0)) and bn,
                    relu=True
                    if (not first or (i != 0)) else False,
                )
            )


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv1d, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              bias=(not bn), **kwargs)
        if bn:
            self.bn = nn.BatchNorm1d(out_channels, momentum=bn_momentum)
        else:
            self.bn = None
        self.relu = relu

        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self):
        init_uniform(self.conv, self.relu)
        if self.bn is not None:
            init_bn(self.bn)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              bias=(not bn), **kwargs)

        if bn:
            self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        else:
            self.bn = None

        self.relu = relu

        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self):
        init_uniform(self.conv, self.relu)
        if self.bn is not None:
            init_bn(self.bn)


class FC(nn.Module):
    def __init__(self, in_channels, out_channels,
                 relu=True, bn=True, bn_momentum=0.1):
        super(FC, self).__init__()

        self.fc = nn.Linear(in_channels, out_channels, bias=(not bn))
        if bn:
            self.bn = nn.BatchNorm1d(out_channels, momentum=bn_momentum)
        else:
            self.bn = None
        self.relu = relu

        self.init_weights()

    def forward(self, x):
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self):
        if self.bn is not None:
            init_bn(self.bn)


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def init_uniform(module, relu=True):
    if module.weight is not None:
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu' if relu else 'linear')
    if module.bias is not None:
        nn.init.zeros_(module.bias)
