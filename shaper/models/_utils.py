import torch
from torch import nn
import torch.nn.functional as F


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv1dBlock, self).__init__()

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


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv2dBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups,
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


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 relu=True, bn=True, bn_momentum=0.1):
        super(LinearBlock, self).__init__()

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
