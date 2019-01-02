import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.nn import MLP
from shaper.nn.functional import encode_one_hot, euclidean_dist

from shaper_compare.models.pointnet import build_pointnet_feature
from shaper_compare.models.pn2ssg import build_pointnet2ssg_feature

_FEATURE_EXTRACTERS = {
    "POINTNET": build_pointnet_feature,
    "PN2SSG": build_pointnet2ssg_feature,
}


class NetworkWrapper(nn.Module):
    def __init__(self, cfg):
        super(NetworkWrapper, self).__init__()
        self.feature_encoder = _FEATURE_EXTRACTERS[cfg.MODEL.TYPE](cfg)
        self.feature_channels = self.feature_encoder.out_channels

        self.class_num = cfg.DATASET.NUM_CLASSES
        self.classifier = nn.Linear(self.feature_encoder.out_channels, self.class_num, bias=True)

        self.class_num_per_batch = cfg.DATASET.PROTO.CLASS_NUM_PER_BATCH
        self.batch_support_num_per_class = cfg.DATASET.PROTO.BATCH_SUPPORT_NUM_PER_CLASS
        self.support_instance_num = cfg.DATASET.PROTO.CLASS_NUM_PER_BATCH * cfg.DATASET.PROTO.BATCH_SUPPORT_NUM_PER_CLASS

        self.init_weights()

    def forward(self, data_batch):
        points = data_batch["points"]
        batch_size = points.size(0)
        target_instance_num = batch_size - self.support_instance_num
        support_points = points[:self.support_instance_num, ...]
        target_points = points[self.support_instance_num:, ...]
        support_features = self.feature_encoder(support_points)
        target_features = self.feature_encoder(target_points)

        support_features = support_features.view(self.class_num_per_batch, self.batch_support_num_per_class,
                                                 self.feature_channels)

        support_centroid_features = torch.mean(support_features, dim=1)

        dist = euclidean_dist(target_features, support_centroid_features)

        proto_preds = F.softmax(-dist, dim=1)  # (target_instance_num, class_num_per_batch)

        support_labels = data_batch["cls_labels"][:self.support_instance_num]
        support_labels = support_labels.view(self.class_num_per_batch, self.batch_support_num_per_class)[:, 0]
        support_labels_one_hot = encode_one_hot(support_labels, self.class_num)

        proto_preds_per_class = torch.mm(proto_preds, support_labels_one_hot)

        direct_preds = self.classifier(target_features)

        preds = {
            'direct_cls_logits': direct_preds,
            'proto_preds': proto_preds,
            'proto_preds_per_class': proto_preds_per_class,
        }

        return preds

    def init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)


class ProtoNetLoss(nn.Module):
    def __init__(self, direct_pred_weight, class_num_per_batch, batch_support_num_per_class):
        super(ProtoNetLoss, self).__init__()
        self.direct_pred_weight = direct_pred_weight
        self.class_num_per_batch = class_num_per_batch
        self.batch_support_num_per_class = batch_support_num_per_class
        self.support_instance_num = class_num_per_batch * batch_support_num_per_class

    def forward(self, preds, data_batch):
        labels = data_batch["cls_labels"]
        batch_size = labels.size(0)
        target_instance_num = batch_size - self.support_instance_num
        support_labels = labels[:self.support_instance_num] \
                             .view(self.class_num_per_batch, self.batch_support_num_per_class)[:, 0]
        target_labels = labels[self.support_instance_num:]

        direct_loss = F.cross_entropy(preds['direct_cls_logits'], target_labels) * self.direct_pred_weight

        proto_preds_log = torch.log(torch.clamp(preds['proto_preds'], 1e-8))
        support_labels_ext = support_labels.unsqueeze(0).expand(target_instance_num, self.class_num_per_batch)
        target_labels_ext = target_labels.unsqueeze(1).expand(target_instance_num, self.class_num_per_batch)

        mask = torch.eq(support_labels_ext, target_labels_ext).type(torch.float)

        proto_loss = -torch.mul(mask, proto_preds_log).mean()

        loss_dict = {
            'proto_loss': proto_loss,
            'direct_loss': direct_loss,
        }

        return loss_dict


class ProtoNetMetric(nn.Module):
    def __init__(self, class_num_per_batch, batch_support_num_per_class, num_classes):
        super(ProtoNetMetric, self).__init__()
        self.class_num_per_batch = class_num_per_batch
        self.batch_support_num_per_class = batch_support_num_per_class
        self.support_instance_num = class_num_per_batch * batch_support_num_per_class

    def forward(self, preds, data_batch):
        labels = data_batch["cls_labels"]
        batch_size = labels.size(0)
        target_instance_num = batch_size - self.support_instance_num
        support_labels = labels[:self.support_instance_num] \
                             .view(self.class_num_per_batch, self.batch_support_num_per_class)[:, 0]
        target_labels = labels[self.support_instance_num:]

        proto_preds = preds['proto_preds'].argmax(1)

        proto_preds_labels = torch.index_select(support_labels, 0, proto_preds)

        acc = proto_preds_labels.eq(target_labels).float()

        return {"acc": acc}


def build_model(cfg):
    net = NetworkWrapper(cfg)

    loss_fn = ProtoNetLoss(
        direct_pred_weight=cfg.PROTO_NET.DIRECT_PRED_WEIGHT,
        class_num_per_batch=cfg.DATASET.PROTO.CLASS_NUM_PER_BATCH,
        batch_support_num_per_class=cfg.DATASET.PROTO.BATCH_SUPPORT_NUM_PER_CLASS,
    )

    metric_fn = ProtoNetMetric(
        class_num_per_batch=cfg.DATASET.PROTO.CLASS_NUM_PER_BATCH,
        batch_support_num_per_class=cfg.DATASET.PROTO.BATCH_SUPPORT_NUM_PER_CLASS,
        num_classes=cfg.DATASET.NUM_CLASSES,
    )

    return net, loss_fn, metric_fn


if __name__ == "__main__":
    from shaper_proto.config import cfg

    cfg.MODEL.TYPE = "POINTNET"
    cfg.DATASET.NUM_CLASSES = 40
    cfg.freeze()

    net, loss_fn, metric_fn = build_model(cfg)
    # net.cuda()

    batch_size = cfg.DATASET.PROTO.CLASS_NUM_PER_BATCH * cfg.DATASET.PROTO.BATCH_SUPPORT_NUM_PER_CLASS + cfg.DATASET.PROTO.TRAIN_BATCH_TARGET_NUM
    num_points = 1024
    in_channels = 3

    data = torch.rand(batch_size, in_channels, num_points)
    cls_labels = torch.randint(cfg.DATASET.NUM_CLASSES, size=(batch_size,))
    data_batch = {"points": data, "cls_labels": cls_labels}
    out_dict = net(data_batch)
    for k, v in out_dict.items():
        print('Prototype Net:', k, v.shape)

    loss = loss_fn(out_dict, data_batch)
