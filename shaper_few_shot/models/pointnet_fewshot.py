import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.models.pointnet import PointNetCls, PointNetClsLoss
from shaper.nn import FC
from shaper.nn.init import set_bn
from shaper.models.metric import Accuracy


class PointNetFewShotCls(PointNetCls):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stem_channels=(64, 64),
                 local_channels=(64, 128, 1024),
                 global_channels=(512, 256),
                 dropout_prob=0.5,
                 with_transform=True,
                 before_classifier_channels=40):
        super(PointNetFewShotCls, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stem_channels=stem_channels,
            local_channels=local_channels,
            global_channels=global_channels,
            dropout_prob=dropout_prob,
            with_transform=with_transform)

        self.dropout_prob = dropout_prob
        self.before_classifier_channels = before_classifier_channels
        if self.before_classifier_channels > 0:
            self.before_classifier = FC(global_channels[-1], before_classifier_channels)
            self.classifier = nn.Linear(before_classifier_channels, out_channels, bias=True)
        else:
            self.classifier = nn.Linear(global_channels[-1], out_channels, bias=True)

        # self.init_weights()
        # set batch normalization to 0.01 as default
        set_bn(self, momentum=0.01)

    def forward(self, data_batch):
        x = data_batch["points"]

        # stem
        x, end_points = self.stem(x)
        # mlp for local features
        x = self.mlp_local(x)
        # max pool over points
        x, max_indices = torch.max(x, 2)
        end_points['key_point_inds'] = max_indices
        # mlp for global features
        x = self.mlp_global(x)
        if self.before_classifier_channels > 0:
            x = self.before_classifier(x)
            # x = F.dropout(x, self.dropout_prob, self.training, inplace=False)
            x = self.classifier(x)
        else:
            x = self.classifier(x)

        preds = {
            'cls_logits': x
        }
        preds.update(end_points)

        return preds


def build_pointnet_fewshot(cfg):
    if cfg.TASK == "classification":
        net = PointNetFewShotCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            stem_channels=cfg.MODEL.POINTNET.STEM_CHANNELS,
            local_channels=cfg.MODEL.POINTNET.LOCAL_CHANNELS,
            global_channels=cfg.MODEL.POINTNET.GLOBAL_CHANNELS,
            dropout_prob=cfg.MODEL.POINTNET.DROPOUT_PROB,
            with_transform=cfg.MODEL.POINTNET.WITH_TRANSFORM,
            before_classifier_channels=cfg.MODEL.POINTNET.BEFORE_CHANNELS,
        )
        loss_fn = PointNetClsLoss(cfg.MODEL.POINTNET.REG_WEIGHT)
        metric_fn = Accuracy()
    else:
        raise NotImplementedError()

    return net, loss_fn, metric_fn


if __name__ == '__main__':
    batch_size = 32
    in_channels = 3
    num_points = 1024
    num_classes = 10

    data = torch.rand(batch_size, in_channels, num_points)

    pointnet = PointNetCls(in_channels, num_classes)
    out_dict = pointnet({"points": data})
    for k, v in out_dict.items():
        print('PointNet:', k, v.shape)
