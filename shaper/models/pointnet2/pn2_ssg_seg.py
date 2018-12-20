"""
point++ with single scale grouping for segmentation
"""

import torch 
import torch.nn as nn

from shaper.nn import MLP, SharedMLP
from shaper.models.point2.modules import PointNetSAModule
from shaper.nn.init import set_bn


