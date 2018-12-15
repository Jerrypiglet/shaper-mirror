import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.models.pn2_ssg import PointNet2SSG_Cls, PointNet2SSG_ClsLoss
from shaper.nn import FC
from shaper.nn.init import set_bn
from shaper.models.metric import Accuracy


class PointNet2SSGFewShotCls(PointNet2SSG_Cls):

    def __init__(self, in_channels, out_channels,
                 num_points_list=(512, 128), radius_list=(0.2, 0.4),
                 nsamples_list=(32, 64), group_mlps=((64, 64, 128), (128, 128, 256)),
                 global_mlps=(256, 512, 1024), fc_channels=(512, 256),
                 drop_prob=0.5, use_xyz=True,
                 before_classifier_channels=40):
        super(PointNet2SSGFewShotCls, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_points_list=num_points_list,
            radius_list=radius_list,
            nsamples_list=nsamples_list,
            group_mlps=group_mlps,
            global_mlps=global_mlps,
            fc_channels=fc_channels,
            drop_prob=drop_prob, use_xyz=use_xyz)

        self.dropout_prob = drop_prob
        self.before_classifier = FC(fc_channels[-1], before_classifier_channels)
        self.classifier = nn.Linear(before_classifier_channels, out_channels, bias=True)

        self.init_weights()
        # set batch normalization to 0.01 as default
        set_bn(self, momentum=0.01)

    def forward(self, data_batch):
        pointcloud = data_batch["points"]
        pointcloud = pointcloud.transpose(1, 2)
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)
            # if xyz is not None:
            #     print('xyz: ', list(xyz.size()))
            # print('features: ', list(features.size()))
        x = self.mlp_global(features.squeeze(-1))
        x = self.before_classifier(x)
        x = F.dropout(x, self.dropout_prob, self.training, inplace=False)
        cls_logits = self.classifier(x)
        preds = {
            'cls_logits': cls_logits
        }

        return preds


def build_pn2ssg_fewshot(cfg):
    if cfg.TASK == "classification":
        net = PointNet2SSGFewShotCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            num_points_list=cfg.MODEL.PN2SSG.NUM_POINTS,
            radius_list=cfg.MODEL.PN2SSG.RADIUS,
            nsamples_list=cfg.MODEL.PN2SSG.NUM_SAMPLE,
            group_mlps=cfg.MODEL.PN2SSG.GROUP_MLPS,
            global_mlps=cfg.MODEL.PN2SSG.GLOBAL_MLPS,
            fc_channels=cfg.MODEL.PN2SSG.FC_CHANNELS,
            drop_prob=cfg.MODEL.PN2SSG.DROP_PROB,
            use_xyz=cfg.MODEL.PN2SSG.USE_XYZ,
            before_classifier_channels=cfg.MODEL.PN2SSG.BEFORE_CHANNELS
        )
        loss_fn = PointNet2SSG_ClsLoss()
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

    pn2ssg = PointNet2SSGFewShotCls(in_channels, num_classes, num_points_list=(128, 64),
                                    nsamples_list=(128, 128))
    pn2ssg.cuda()
    out_dict = pn2ssg({"points": data})
    for k, v in out_dict.items():
        print('pointnet2ssg fewshot:', k, v.shape)
