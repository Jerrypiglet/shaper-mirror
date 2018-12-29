import torch
import torch.nn as nn

from shaper.models.pointnet.pointnet_cls import PointNetCls, PointNetClsLoss
from shaper.nn import FC
from shaper.nn.init import set_bn
from shaper.models.metric import ClsAccuracy


class PointNetFewShotCls(PointNetCls):
    def __init__(self,
                 *args,
                 penult_channels=0,
                 penult_dropout=False,
                 **kwargs,
                 ):
        global_channels = kwargs["global_channels"]
        out_channels = kwargs["out_channels"]
        dropout_prob = kwargs["dropout_prob"]
        super(PointNetFewShotCls, self).__init__(*args, **kwargs)

        if penult_channels > 0:
            self.penult_classifier = FC(global_channels[-1], penult_channels)
            self.penult_dropout = nn.Dropout(p=dropout_prob) if penult_dropout else None
            self.classifier = nn.Linear(penult_channels, out_channels, bias=True)
        else:
            self.penult_classifier = None
            self.penult_dropout = None
            self.classifier = nn.Linear(global_channels[-1], out_channels, bias=True)

        self.init_weights()
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
        end_points["key_point_inds"] = max_indices

        # mlp for global features
        x = self.mlp_global(x)
        end_points["cls_feature"] = x

        if self.penult_classifier is not None:
            x = self.penult_classifier(x)
        if self.penult_dropout is not None:
            x = self.penult_dropout(x)
        x = self.classifier(x)

        preds = {
            "cls_logit": x
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
            penult_channels=cfg.MODEL.PENULT_CHANNELS,
        )
        loss_fn = PointNetClsLoss(cfg.MODEL.POINTNET.REG_WEIGHT)
        metric_fn = ClsAccuracy()
    else:
        raise NotImplementedError()

    return net, loss_fn, metric_fn


if __name__ == '__main__':
    batch_size = 32
    in_channels = 3
    num_points = 1024
    num_classes = 10
    global_channels = (512, 256)

    data = torch.rand(batch_size, in_channels, num_points)

    pointnet = PointNetFewShotCls(in_channels,
                                  out_channels=num_classes,
                                  global_channels=global_channels,
                                  dropout_prob=0.3)
    out_dict = pointnet({"points": data})
    for k, v in out_dict.items():
        print('PointNet:', k, v.shape)
