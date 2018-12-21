import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.models.pn2_ssg import PointNet2SSGCls, ClsLoss
from shaper.nn import FC
from shaper.nn.init import set_bn
from shaper.models.metric import Accuracy


class PointNet2SSGFewShotCls(PointNet2SSGCls):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_centroids=(512, 128),
                 radius=(0.2, 0.4),
                 num_neighbours=(32, 64),
                 sa_channels=((64, 64, 128), (128, 128, 256)),
                 local_channels=(256, 512, 1024),
                 global_channels=(512, 256),
                 dropout_prob=0.5,
                 use_xyz=True,
                 before_classifier_channels=40):
        super(PointNet2SSGFewShotCls, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_centroids=num_centroids,
            radius=radius,
            num_neighbours=num_neighbours,
            sa_channels=sa_channels,
            local_channels=local_channels,
            global_channels=global_channels,
            dropout_prob=dropout_prob,
            use_xyz=use_xyz)

        self.dropout_prob = dropout_prob
        self.before_classifier_channels = before_classifier_channels
        if self.before_classifier_channels > 0:
            self.before_classifier = FC(global_channels[-1], before_classifier_channels)
            self.classifier = nn.Linear(before_classifier_channels, out_channels, bias=True)
        else:
            self.classifier = nn.Linear(global_channels[-1], out_channels, bias=True)

        self.init_weights()
        # set batch normalization to 0.01 as default
        set_bn(self, momentum=0.01)

    def forward(self, data_batch):
        point = data_batch["points"]
        end_points = {}

        # torch.Tensor.narrow; share same memory
        xyz = point.narrow(1, 0, 3)
        if point.size(1) > 3:
            feature = point.narrow(1, 3, point.size(1) - 3)
        else:
            feature = None

        for sa_module in self.sa_modules:
            xyz, feature = sa_module(xyz, feature)

        if self.use_xyz:
            x = torch.cat([xyz, feature], dim=1)
        else:
            x = feature
        x = self.mlp_local(x)
        x, max_indices = torch.max(x, 2)
        end_points['key_point_inds'] = max_indices
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


def build_pn2ssg_fewshot(cfg):
    if cfg.TASK == "classification":
        net = PointNet2SSGFewShotCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            num_centroids=cfg.MODEL.PN2SSG.NUM_CENTROIDS,
            radius=cfg.MODEL.PN2SSG.RADIUS,
            num_neighbours=cfg.MODEL.PN2SSG.NUM_NEIGHBOURS,
            sa_channels=cfg.MODEL.PN2SSG.SA_CHANNELS,
            local_channels=cfg.MODEL.PN2SSG.LOCAL_CHANNELS,
            global_channels=cfg.MODEL.PN2SSG.GLOBAL_CHANNELS,
            dropout_prob=cfg.MODEL.PN2SSG.DROPOUT_PROB,
            use_xyz=cfg.MODEL.PN2SSG.USE_XYZ,
            before_classifier_channels=cfg.MODEL.PN2SSG.BEFORE_CHANNELS
        )
        loss_fn = ClsLoss()
        metric_fn = Accuracy()
    else:
        raise NotImplementedError

    return net, loss_fn, metric_fn


if __name__ == "__main__":
    batch_size = 8
    in_channels = 3
    num_points = 1024
    num_classes = 40

    data = torch.rand(batch_size, in_channels, num_points)
    data = data.cuda()

    pn2ssg = PointNet2SSGFewShotCls(in_channels, num_classes, num_centroids=(128, 64),
                                    num_neighbours=(128, 128))
    pn2ssg.cuda()
    out_dict = pn2ssg({"points": data})
    for k, v in out_dict.items():
        print('pointnet2ssg fewshot:', k, v.shape)
