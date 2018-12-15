import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.models.dgcnn import DGCNNCls, ClsLoss
from shaper.nn import FC
from shaper.nn.init import set_bn
from shaper.models.metric import Accuracy


# from shaper.models.dgcnn_utils import get_edge_feature


class DGCNNFewShotCls(DGCNNCls):

    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_conv_channels=(64, 64, 64, 128),
                 inter_channels=1024,
                 global_channels=(512, 256),
                 k=20,
                 dropout_prob=0.5,
                 with_transform=True,
                 before_classifier_channels=40):

        super(DGCNNFewShotCls, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            edge_conv_channels=edge_conv_channels,
            inter_channels=inter_channels,
            global_channels=global_channels,
            k=k,
            dropout_prob=dropout_prob,
            with_transform=with_transform)

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
        end_points = {}
        x = data_batch["points"]
        # input transform
        if self.with_transform:
            trans_input = self.transform_input(x)
            x = torch.bmm(trans_input, x)
            end_points['trans_input'] = trans_input

        # EdgeConvMLP
        features = []
        for edge_conv in self.mlp_edge_conv:
            # x = get_edge_feature(x, self.k)
            x = edge_conv(x)
            # x, _ = torch.max(x, 3)
            features.append(x)

        x = torch.cat(features, dim=1)

        x = self.mlp_local(x)
        x, max_indices = torch.max(x, 2)
        end_points['key_point_inds'] = max_indices
        x = self.mlp_global(x)
        if self.before_classifier_channels > 0:
            x = self.before_classifier(x)
            x = F.dropout(x, self.dropout_prob, self.training, inplace=False)
            x = self.classifier(x)
        else:
            x = self.classifier(x)
        preds = {
            'cls_logits': x
        }
        preds.update(end_points)

        return preds


def build_dgcnn_fewshot(cfg):
    if cfg.TASK == "classification":
        net = DGCNNFewShotCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            edge_conv_channels=cfg.MODEL.DGCNN.EDGE_CONV_CHANNELS,
            inter_channels=cfg.MODEL.DGCNN.INTER_CHANNELS,
            global_channels=cfg.MODEL.DGCNN.GLOBAL_CHANNELS,
            k=cfg.MODEL.DGCNN.K,
            dropout_prob=cfg.MODEL.DGCNN.DROPOUT_PROB,
            before_classifier_channels=cfg.MODEL.DGCNN.BEFORE_CHANNELS
        )
        loss_fn = ClsLoss(cfg.MODEL.DGCNN.LABEL_SMOOTHING)
        metric_fn = Accuracy()
    else:
        raise NotImplementedError()

    return net, loss_fn, metric_fn


if __name__ == "__main__":
    batch_size = 4
    in_channels = 3
    num_points = 1024
    num_classes = 10

    data = torch.rand(batch_size, in_channels, num_points).cuda()

    dgcnn = DGCNNFewShotCls(in_channels, num_classes, with_transform=False).cuda()
    out_dict = dgcnn({"points": data})
    for k, v in out_dict.items():
        print('DGCNN:', k, v.shape)
