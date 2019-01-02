import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.models.pn2_s2cnn import PN2S2CNNCls, PNS2CNNClsLoss, TNet, _F
from shaper.nn import FC
from shaper.nn.init import set_bn
from shaper.models.metric import Accuracy


class PN2S2CNNFewShotCls(PN2S2CNNCls):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_centroids=16,
                 radius_list=(0.2,),
                 num_neighbours_list=(16,),
                 band_width_in_list=(16,),
                 s2cnn_feature_channels_list=((32, 64),),
                 band_width_list=((16, 8),),
                 k=4,
                 local_channels=(256, 512, 1024),
                 global_channels=(512, 256),
                 dropout_prob=0.5,
                 transform_xyz=True,
                 before_classifier_channels=40
                 ):
        super(PN2S2CNNFewShotCls, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_centroids=num_centroids,
            radius_list=radius_list,
            num_neighbours_list=num_neighbours_list,
            band_width_in_list=band_width_in_list,
            s2cnn_feature_channels_list=s2cnn_feature_channels_list,
            band_width_list=band_width_list,
            k=k,
            local_channels=local_channels,
            global_channels=global_channels,
            dropout_prob=dropout_prob,
            transform_xyz=transform_xyz,
        )

        self.before_classifier_channels = before_classifier_channels
        if self.before_classifier_channels > 0:
            self.before_classifier = FC(global_channels[-1], before_classifier_channels)
            self.classifier = nn.Linear(before_classifier_channels, out_channels, bias=True)
        else:
            self.classifier = nn.Linear(global_channels[-1], out_channels, bias=True)

        set_bn(self, momentum=0.01)

    def forward(self, data_batch):
        end_points = {}
        point = data_batch["points"]
        xyz = point.narrow(1, 0, 3).contiguous()  # [b, 3, n]
        # xyz_flipped = xyz.transpose(1, 2).contiguous()  # [b, n, 3]
        if self.use_normal:
            features = point.narrow(1, 3, 6).contiguous()
        else:
            features = None
        if point.size(2) == self.num_centroids:
            new_xyz = xyz
        else:
            index = self.sampler(xyz)
            new_xyz = _F.gather_points(xyz, index)

        batch_size = point.size(0)

        local_s2cnn_features_list = []
        for i in range(self.local_group_scale_num):
            new_features, pts_cnt = self.groupers[i](
                new_xyz, xyz, features)  # new_features: [b, c, nc, nn], pts_cnt: [b, nc]

            new_features = new_features.transpose(1, 2).contiguous()
            new_features = torch.div(new_features, self.radius_list[i])
            new_features = new_features.view(batch_size * self.num_centroids, self.in_channels,
                                             self.num_neighbours_list[i])

            pts_cnt = pts_cnt.view(batch_size * self.num_centroids)
            local_s2cnn_features = self.local_s2cnn_list[i](new_features, pts_cnt)
            local_s2cnn_features = local_s2cnn_features.view(batch_size, self.num_centroids, -1)
            local_s2cnn_features = local_s2cnn_features.transpose(1, 2).contiguous()

            local_s2cnn_features_list.append(local_s2cnn_features)

        local_s2cnn_features_list.append(new_xyz)
        x = torch.cat(local_s2cnn_features_list, dim=1)

        if self.transform_xyz:
            trans_matrix_xyz = self.transform_input(x)
            new_xyz = torch.bmm(trans_matrix_xyz, new_xyz)
            end_points["trans_matrix_xyz"] = trans_matrix_xyz
            local_s2cnn_features_list.append(new_xyz)
            x = torch.cat(local_s2cnn_features_list, dim=1)

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


def build_pns2cnn_fewshot(cfg):
    if cfg.TASK == "classification":
        net = PN2S2CNNFewShotCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            num_centroids=cfg.MODEL.PNS2CNN.NUM_CENTROIDS,
            radius_list=cfg.MODEL.PNS2CNN.RADIUS_LIST,
            num_neighbours_list=cfg.MODEL.PNS2CNN.NUM_NEIGHBOURS_LIST,
            band_width_in_list=cfg.MODEL.PNS2CNN.BAND_WIDTH_IN_LIST,
            s2cnn_feature_channels_list=cfg.MODEL.PNS2CNN.FEATURE_CHANNELS_LIST,
            band_width_list=cfg.MODEL.PNS2CNN.BAND_WIDTH_LIST,
            k=cfg.MODEL.PNS2CNN.K,
            local_channels=cfg.MODEL.PNS2CNN.LOCAL_CHANNELS,
            global_channels=cfg.MODEL.PNS2CNN.GLOBAL_CHANNELS,
            dropout_prob=cfg.MODEL.PNS2CNN.DROPOUT_PROB,
            transform_xyz=cfg.MODEL.PNS2CNN.TRANSFORM_XYZ,
            before_classifier_channels=cfg.MODEL.PNS2CNN.BEFORE_CHANNELS,
        )

        loss_fn = PNS2CNNClsLoss(
            trans_reg_weight=cfg.MODEL.PNS2CNN.TRANS_REG_WEIGHT
        )
        metric_fn = Accuracy()

    else:
        raise NotImplementedError

    return net, loss_fn, metric_fn


if __name__ == "__main__":
    batch_size = 2
    in_channels = 3
    num_points = 1024
    num_classes = 10

    data = torch.rand(batch_size, in_channels, num_points)
    data = data.cuda()

    pns2cnn = PN2S2CNNFewShotCls(3, num_classes)
    pns2cnn = pns2cnn.cuda()

    out_dict = pns2cnn({"points": data})
    for k, v in out_dict.items():
        print("pns2cnn: ", k, v.size())
