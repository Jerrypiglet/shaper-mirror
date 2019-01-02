import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.nn import MLP
from shaper.nn.functional import encode_one_hot

from shaper_compare.models.pointnet import build_pointnet_feature
from shaper_compare.models.pn2ssg import build_pointnet2ssg_feature

_FEATURE_EXTRACTERS = {
    "POINTNET": build_pointnet_feature,
    "PN2SSG": build_pointnet2ssg_feature,
}


class CompareNet(nn.Module):
    def __init__(self, in_channels, mlp_channels,
                 drop_prob=0.0):
        super(CompareNet, self).__init__()
        self.mlp = MLP(in_channels, mlp_channels, drop_prob)
        self.linear = nn.Linear(mlp_channels[-1], 1)
        self.init_weights()

    def forward(self, x):
        x = self.mlp(x)
        x = self.linear(x)
        x = torch.sigmoid(x)

        return x

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


class NetworkWrapper(nn.Module):
    def __init__(self, cfg):
        super(NetworkWrapper, self).__init__()
        self.feature_encoder = _FEATURE_EXTRACTERS[cfg.MODEL.TYPE](cfg)
        self.feature_channels = self.feature_encoder.out_channels

        self.compare_net = CompareNet(in_channels=self.feature_channels * 2,
                                      mlp_channels=cfg.COMPARE_NET.MLPS,
                                      drop_prob=cfg.COMPARE_NET.DROP_PROB)

        self.class_num = cfg.DATASET.NUM_CLASSES
        self.classifier = nn.Linear(self.feature_encoder.out_channels, self.class_num, bias=True)

        self.support_instance_num = cfg.DATASET.COMPARE.CLASS_NUM_PER_BATCH * cfg.DATASET.COMPARE.BATCH_SUPPORT_NUM_PER_CLASS
        # self.target_instance_num = cfg.DATASET.COMPARE.BATCH_TARGET_NUM

        self.init_weights()

    def forward(self, data_batch):
        points = data_batch["points"]
        batch_size = points.size(0)
        target_instance_num = batch_size - self.support_instance_num
        support_points = points[:self.support_instance_num, ...]
        target_points = points[self.support_instance_num:, ...]
        support_features = self.feature_encoder(support_points)
        target_features = self.feature_encoder(target_points)

        support_features_ext = support_features.unsqueeze(1).repeat(1, target_instance_num, 1)
        target_features_ext = target_features.unsqueeze(0).repeat(self.support_instance_num, 1, 1)

        relation_pairs = torch.cat((support_features_ext, target_features_ext), -1).view(-1, self.feature_channels * 2)
        relations = self.compare_net(relation_pairs).view(-1)

        relation_preds_T = relations.view(self.support_instance_num, target_instance_num).transpose(0, 1)  # (tar, sup)

        support_labels = data_batch["cls_labels"][:self.support_instance_num]
        support_labels_one_hot = encode_one_hot(support_labels, self.class_num)

        relation_preds_per_class = torch.mm(relation_preds_T, support_labels_one_hot)

        direct_preds = self.classifier(target_features)

        preds = {
            'direct_cls_logits': direct_preds,
            'relation_preds': relations,
            'relation_preds_per_class': relation_preds_per_class,
        }

        return preds

    def init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)


class CompareNetLoss(nn.Module):
    def __init__(self, direct_pred_weight, support_instance_num):
        super(CompareNetLoss, self).__init__()
        self.direct_pred_weight = direct_pred_weight
        self.support_instance_num = support_instance_num
        # self.target_instance_num = target_instance_num

        self.compare_loss = nn.MSELoss(reduction="sum")

    def forward(self, preds, data_batch):
        labels = data_batch["cls_labels"]
        batch_size = labels.size(0)
        target_instance_num = batch_size - self.support_instance_num
        support_labels = labels[:self.support_instance_num]
        target_labels = labels[self.support_instance_num:]
        support_labels_ext = support_labels.unsqueeze(1).repeat(1, target_instance_num).view(-1)
        target_labels_ext = target_labels.unsqueeze(0).repeat(self.support_instance_num, 1).view(-1)

        compare_onehot_labels = torch.eq(support_labels_ext, target_labels_ext).type(torch.float)
        compare_loss = self.compare_loss(preds['relation_preds'], compare_onehot_labels) / float(target_instance_num)

        direct_loss = F.cross_entropy(preds['direct_cls_logits'], target_labels) * self.direct_pred_weight

        loss_dict = {
            'compare_loss': compare_loss,
            'direct_loss': direct_loss,
        }

        return loss_dict


class CompareNetMetric(nn.Module):
    def __init__(self, support_instance_num, num_classes):
        super(CompareNetMetric, self).__init__()
        self.support_instance_num = support_instance_num
        self.num_classes = num_classes

    def forward(self, preds, data_batch):
        labels = data_batch["cls_labels"]
        batch_size = labels.size(0)
        target_instance_num = batch_size - self.support_instance_num
        support_labels = labels[:self.support_instance_num]
        target_labels = labels[self.support_instance_num:]
        # support_labels_ext = support_labels.unsqueeze(1).repeat(1, self.target_instance_num).view(-1)
        # target_labels_ext = target_labels.unsqueeze(0).repeat(self.support_instance_num, 1).view(-1)

        relation_preds = preds['relation_preds'].view(self.support_instance_num, target_instance_num).transpose(0, 1)

        relation_preds = relation_preds.argmax(1)

        relation_pred_labels = torch.index_select(support_labels, 0, relation_preds)

        acc = relation_pred_labels.eq(target_labels).float()

        return {"acc": acc}


def build_model(cfg):
    net = NetworkWrapper(cfg)

    loss_fn = CompareNetLoss(cfg.COMPARE_NET.DIRECT_PRED_WEIGHT,
                             support_instance_num=cfg.DATASET.COMPARE.CLASS_NUM_PER_BATCH * cfg.DATASET.COMPARE.BATCH_SUPPORT_NUM_PER_CLASS,
                             )

    metric_fn = CompareNetMetric(
        support_instance_num=cfg.DATASET.COMPARE.CLASS_NUM_PER_BATCH * cfg.DATASET.COMPARE.BATCH_SUPPORT_NUM_PER_CLASS,
        num_classes=cfg.DATASET.NUM_CLASSES)

    return net, loss_fn, metric_fn


if __name__ == "__main__":
    from shaper_compare.config import cfg

    cfg.MODEL.TYPE = "PointNet"
    cfg.DATASET.NUM_CLASSES = 40
    cfg.freeze()

    net, loss_fn, metric_fn = build_model(cfg)

    batch_size = cfg.DATASET.COMPARE.CLASS_NUM_PER_BATCH * cfg.DATASET.COMPARE.BATCH_SUPPORT_NUM_PER_CLASS + cfg.DATASET.COMPARE.TRAIN_BATCH_TARGET_NUM
    num_points = 1024
    in_channels = 3

    data = torch.rand(batch_size, in_channels, num_points)
    out_dict = net({"points": data})
    for k, v in out_dict.items():
        print('Compare Net:', k, v.shape)
