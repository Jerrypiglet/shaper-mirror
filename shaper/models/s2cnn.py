"""S2CNN

References:
    @inproceedings{s2cnn,
        title={Spherical {CNN}s},
        author={Taco S. Cohen and Mario Geiger and Jonas Köhler and Max Welling},
        booktitle={International Conference on Learning Representations},
        year={2018},
        url={https://openreview.net/forum?id=Hkbd5xZRb},
    }
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.models.metric import Accuracy
from shaper.models.s2cnn_modules import S2Convolution, SO3Convolution
from shaper.models.s2cnn_modules import s2_equatorial_grid, so3_equatorial_grid, so3_integrate
from shaper.models.s2cnn_modules.utils.pc2sph import PointCloudProjector


class S2CNNCls(nn.Module):
    def __init__(self, in_channels, out_channels,
                 band_width_in=30,
                 feature_channels=(100, 100),
                 band_width_list=(16, 10)):
        super(S2CNNCls, self).__init__()
        assert (in_channels in [3, 6])

        self.feature_channels = feature_channels

        self.use_normal = False
        if in_channels == 6:
            self.use_normal = True

        self.pc_projector = PointCloudProjector(band_width_in)

        # xyz layers
        xyz_sequence = []
        xyz_grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * band_width_in, n_beta=1)
        xyz_sequence.append(S2Convolution(1, feature_channels[0], band_width_in, band_width_list[0], xyz_grid))

        # xyz SO3 layers
        for l in range(len(feature_channels) - 1):
            nfeature_in = self.feature_channels[l]
            nfeature_out = self.feature_channels[l + 1]
            b_in = band_width_list[l]
            b_out = band_width_list[l + 1]

            xyz_sequence.append(nn.BatchNorm3d(nfeature_in, affine=True))
            xyz_sequence.append(nn.ReLU())
            grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2 * b_in, n_beta=1, n_gamma=1)
            xyz_sequence.append(SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, grid))

        xyz_sequence.append(nn.BatchNorm3d(nfeature_out, affine=True))
        xyz_sequence.append(nn.ReLU())

        self.xyz_sequential = nn.Sequential(*xyz_sequence)

        # normal layers
        if self.use_normal:
            norm_sequence = []
            norm_grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * band_width_in, n_beta=1)
            norm_sequence.append(S2Convolution(1, feature_channels[0], band_width_in, band_width_list[0], norm_grid))

            # norm SO3 layers
            for l in range(len(feature_channels) - 1):
                nfeature_in = self.feature_channels[l]
                nfeature_out = self.feature_channels[l + 1]
                b_in = band_width_list[l]
                b_out = band_width_list[l + 1]

                norm_sequence.append(nn.BatchNorm3d(nfeature_in, affine=True))
                norm_sequence.append(nn.ReLU())
                grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2 * b_in, n_beta=1, n_gamma=1)
                norm_sequence.append(SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, grid))

            norm_sequence.append(nn.BatchNorm3d(nfeature_out, affine=True))
            norm_sequence.append(nn.ReLU())

            self.norm_sequential = nn.Sequential(*norm_sequence)

        if self.use_normal:
            self.linear = nn.Linear(nfeature_out * 2, out_channels)
        else:
            self.linear = nn.Linear(nfeature_out, out_channels)

    def forward(self, x):
        if self.use_normal:
            xyz, norm = self.pc_projector(x)
            xyz = self.xyz_sequential(xyz)
            xyz = so3_integrate(xyz)
            norm = self.norm_sequential(norm)
            norm = so3_integrate(norm)
            inter_feature = torch.cat((xyz, norm), -1)
        else:
            xyz = self.pc_projector(x)
            xyz = self.xyz_sequential(xyz)
            xyz = so3_integrate(xyz)
            inter_feature = xyz

        cls_logits = self.linear(inter_feature)
        preds = {
            'cls_logits': cls_logits
        }

        return preds

    def init_weights(self):
        nn.init.kaiming_uniform_(self.out_layer.weight, nonlinearity='linear')
        nn.init.zeros_(self.out_layer.bias)


class S2CNNFeature(nn.Module):
    """Spherical cnn for feature extraction."""

    def __init__(self, in_channels, band_width_in=30,
                 feature_channels=(100, 100),
                 band_width_list=(16, 10)):
        super(S2CNNFeature, self).__init__()
        assert (in_channels in [3, 6])

        self.feature_channels = feature_channels

        self.use_normal = False
        self.out_channels = feature_channels[-1]
        if in_channels == 6:
            self.use_normal = True
            self.out_channels = 2 * feature_channels[-1]

        self.pc_projector = PointCloudProjector(band_width_in)

        # xyz layers
        xyz_sequence = []
        xyz_grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * band_width_in, n_beta=1)
        xyz_sequence.append(S2Convolution(1, feature_channels[0], band_width_in, band_width_list[0], xyz_grid))

        # xyz SO3 layers
        for l in range(len(feature_channels) - 1):
            nfeature_in = self.feature_channels[l]
        nfeature_out = self.feature_channels[l + 1]
        b_in = band_width_list[l]
        b_out = band_width_list[l + 1]

        xyz_sequence.append(nn.BatchNorm3d(nfeature_in, affine=True))
        xyz_sequence.append(nn.ReLU())
        grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2 * b_in, n_beta=1, n_gamma=1)
        xyz_sequence.append(SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, grid))

        xyz_sequence.append(nn.BatchNorm3d(nfeature_out, affine=True))
        xyz_sequence.append(nn.ReLU())

        self.xyz_sequential = nn.Sequential(*xyz_sequence)

        # normal layers
        if self.use_normal:
            norm_sequence = []
            norm_grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * band_width_in, n_beta=1)
            norm_sequence.append(S2Convolution(1, feature_channels[0], band_width_in, band_width_list[0], norm_grid))

            # norm SO3 layers
            for l in range(len(feature_channels) - 1):
                nfeature_in = self.feature_channels[l]
            nfeature_out = self.feature_channels[l + 1]
            b_in = band_width_list[l]
            b_out = band_width_list[l + 1]

            norm_sequence.append(nn.BatchNorm3d(nfeature_in, affine=True))
            norm_sequence.append(nn.ReLU())
            grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2 * b_in, n_beta=1, n_gamma=1)
            norm_sequence.append(SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, grid))

            norm_sequence.append(nn.BatchNorm3d(nfeature_out, affine=True))
            norm_sequence.append(nn.ReLU())

            self.norm_sequential = nn.Sequential(*norm_sequence)

    def forward(self, x):
        if self.use_normal:
            xyz, norm = self.pc_projector(x)
            xyz = self.xyz_sequential(xyz)
            xyz = so3_integrate(xyz)
            norm = self.norm_sequential(norm)
            norm = so3_integrate(norm)
            inter_feature = torch.cat((xyz, norm), -1)
        else:
            xyz = self.pc_projector(x)
            xyz = self.xyz_sequential(xyz)
            xyz = so3_integrate(xyz)
            inter_feature = xyz

        return inter_feature


class S2CNNClsLoss(nn.Module):
    def __init__(self):
        super(S2CNNClsLoss, self).__init__()

    def forward(self, preds, labels):
        cls_logits = preds["cls_logits"]
        cls_labels = labels["cls_labels"]
        cls_loss = F.cross_entropy(cls_logits, cls_labels)
        loss_dict = {
            'cls_loss': cls_loss,
        }
        return loss_dict


def build_s2cnn(cfg):
    if cfg.TASK == "classification":
        net = S2CNNCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            band_width_in=cfg.MODEL.S2CNN.BAND_WIDTH_IN,
            feature_channels=cfg.MODEL.S2CNN.FEATURE_CHANNELS,
            band_width_list=cfg.MODEL.S2CNN.BAND_WIDTH_LIST,
        )
        loss_fn = S2CNNClsLoss()
        metric_fn = Accuracy()
    else:
        raise NotImplementedError()

    return net, loss_fn, metric_fn


if __name__ == "__main__":
    x = np.random.rand(4, 6, 1024)
    x = torch.as_tensor(x).type(torch.float32)
    s2cnn = S2CNNCls(6, 10)
    pred_x = s2cnn(x)
    print("pred_x: \n", pred_x['cls_logits'].size())
