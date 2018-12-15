import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.models.pn2_msg import Pointnet2MSG_Cls, PointNet2MSG_ClsLoss
from shaper.nn import FC
from shaper.nn.init import set_bn
from shaper.models.metric import Accuracy


class PointNet2MSGFewShotCls(Pointnet2MSG_Cls):

    def __init__(self, input_channels, out_channels,
                 num_points_list=(512, 128), radius_list=((0.1, 0.2, 0.4), (0.2, 0.4, 0.8)),
                 nsamples_list=((16, 32, 128), (32, 64, 128)),
                 group_mlps_list=(
                         ((32, 32, 64), (64, 64, 128), (64, 96, 128)),
                         ((64, 64, 128), (128, 128, 256), (128, 128, 256))),
                 global_mlps=(256, 512, 1024), fc_channels=(512, 256), drop_prob=0.5,
                 use_xyz=True,
                 before_classifier_channels=40):
        super(PointNet2MSGFewShotCls, self).__init__(
            input_channels=input_channels,
            out_channels=out_channels,
            num_points_list=num_points_list,
            radius_list=radius_list,
            nsamples_list=nsamples_list,
            group_mlps_list=group_mlps_list,
            global_mlps=global_mlps,
            fc_channels=fc_channels,
            drop_prob=drop_prob, use_xyz=use_xyz)

        self.dropout_prob = drop_prob
        self.before_classifier = FC(fc_channels[-1], before_classifier_channels)
        self.classifier = nn.Linear(before_classifier_channels, out_channels)

        self.init_weights()
        # set batch normalization to 0.01 as default
        set_bn(self, momentum=0.01)

    def forward(self, data_batch):
        pointcloud = data_batch["points"]
        pointcloud = pointcloud.transpose(1, 2)
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        x = self.mlp_global(features.squeeze(-1))
        x = self.before_classifier(x)
        x = F.dropout(x, self.dropout_prob, self.training, inplace=False)
        cls_logits = self.classifier(x)
        preds = {
            'cls_logits': cls_logits
        }

        return preds


def build_pn2msg_fewshot(cfg):
    if cfg.TASK == "classification":
        net = PointNet2MSGFewShotCls(
            input_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            num_points_list=cfg.MODEL.PN2MSG.NUM_POINTS,
            radius_list=cfg.MODEL.PN2MSG.RADIUS,
            nsamples_list=cfg.MODEL.PN2MSG.NUM_SAMPLE,
            group_mlps_list=cfg.MODEL.PN2MSG.GROUP_MLPS,
            global_mlps=cfg.MODEL.PN2MSG.GLOBAL_MLPS,
            fc_channels=cfg.MODEL.PN2MSG.FC_CHANNELS,
            drop_prob=cfg.MODEL.PN2MSG.DROP_PROB,
            use_xyz=cfg.MODEL.PN2MSG.USE_XYZ,
            before_classifier_channels=cfg.MODEL.PN2MSG.BEFORE_CHANNELS
        )
        loss_fn = PointNet2MSG_ClsLoss()
        metric_fn = Accuracy()
    else:
        raise NotImplementedError

    return net, loss_fn, metric_fn


if __name__ == '__main__':
    batch_size = 8
    in_channels = 3
    num_points = 1024
    num_classes = 40

    data = torch.rand(batch_size, in_channels, num_points)
    data = data.cuda()

    pn2msg = PointNet2MSGFewShotCls(3, 40)
    pn2msg.cuda()
    out_dict = pn2msg({"points": data})
    for k, v in out_dict.items():
        print('pointnet2msg fewshot:', k, v.shape)
