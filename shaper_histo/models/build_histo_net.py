import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from numpy.testing import assert_almost_equal

from shaper.nn import MLP
from shaper.nn.functional import encode_one_hot

from shaper_compare.models.pointnet import build_pointnet_feature
from shaper_compare.models.pn2ssg import build_pointnet2ssg_feature
from shaper_histo.utils import L2Normalization

_FEATURE_EXTRACTERS = {
    "POINTNET": build_pointnet_feature,
    "PN2SSG": build_pointnet2ssg_feature,
}


class NetworkWrapper(nn.Module):
    def __init__(self, cfg):
        super(NetworkWrapper, self).__init__()
        self.mode = cfg.HISTO_NET.MODE
        assert (self.mode in ["max", "sum"]), "Undefined mode: {}".format(self.mode)
        self.feature_encoder = _FEATURE_EXTRACTERS[cfg.MODEL.TYPE](cfg)
        self.feature_channels = self.feature_encoder.out_channels

        self.l2_normalizer = L2Normalization()

        self.class_num = cfg.DATASET.NUM_CLASSES
        self.classifier = nn.Linear(self.feature_encoder.out_channels, self.class_num, bias=True)

        self.class_num_per_batch = cfg.DATASET.HISTO.CLASS_NUM_PER_BATCH
        self.batch_support_num_per_class = cfg.DATASET.HISTO.BATCH_SUPPORT_NUM_PER_CLASS
        self.support_instance_num = cfg.DATASET.HISTO.CLASS_NUM_PER_BATCH * cfg.DATASET.HISTO.BATCH_SUPPORT_NUM_PER_CLASS

        self.init_weights()

    def forward(self, data_batch):
        points = data_batch["points"]
        batch_size = points.size(0)
        target_instance_num = batch_size - self.support_instance_num
        support_points = points[:self.support_instance_num, ...]
        target_points = points[self.support_instance_num:, ...]
        support_features = self.feature_encoder(support_points)
        support_features = self.l2_normalizer(support_features)
        target_features = self.feature_encoder(target_points)
        target_features = self.l2_normalizer(target_features)

        support_features_ext = support_features.unsqueeze(0).expand(target_instance_num, self.support_instance_num,
                                                                    self.feature_channels).unsqueeze(-1)

        target_features_ext = target_features.unsqueeze(1).expand(target_instance_num, self.support_instance_num,
                                                                  self.feature_channels).unsqueeze(2)

        inner_product = torch.matmul(target_features_ext, support_features_ext).squeeze_(-1).squeeze_(-1)

        preds = {}
        preds['inner_product'] = inner_product
        support_labels = data_batch["cls_labels"][:self.support_instance_num]
        if self.mode == "max":
            inner_product_preds = inner_product.argmax(1)
            inner_product_preds = torch.index_select(support_labels, 0, inner_product_preds)
        else:
            inner_product = inner_product.view(target_instance_num, self.class_num_per_batch,
                                               self.batch_support_num_per_class)
            inner_product = torch.mean(inner_product, dim=-1)

            support_labels = support_labels.view(self.class_num_per_batch, self.batch_support_num_per_class)[:, 0]
            inner_product_preds = inner_product.argmax(1)
            inner_product_preds = torch.index_select(support_labels, 0, inner_product_preds)

        preds['inner_product_preds'] = inner_product_preds

        direct_preds = self.classifier(target_features)
        preds['direct_cls_logits'] = direct_preds

        return preds

    def init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)


class HistoNetLoss(nn.Module):
    def __init__(self, num_steps, direct_pred_weight, class_num_per_batch, batch_support_num_per_class):
        super(HistoNetLoss, self).__init__()
        self.step = 2 / (num_steps - 1)
        self.t = torch.arange(-1, 1 + self.step, self.step).view(-1, 1)
        self.tsize = self.t.size()[0]

        self.direct_pred_weight = direct_pred_weight
        self.class_num_per_batch = class_num_per_batch
        self.batch_support_num_per_class = batch_support_num_per_class
        self.support_instance_num = class_num_per_batch * batch_support_num_per_class

    def forward(self, preds, data_batch):
        def histogram(inds, size):
            s_repeat_ = s_repeat.clone()
            # indsa = (torch.eq(s_repeat_floor, self.t.repeat(1, s_repeat_floor.size(1)))) & inds
            indsa = torch.eq(s_repeat_floor, self.t.repeat(1, s_repeat_floor.size(1)))
            # indsa1 = indsa.cpu().numpy()
            # import numpy as np
            # np.savetxt("indsa1.txt", indsa1, fmt="%d")
            indsa = indsa & inds
            # indsa2 = indsa.cpu().numpy()
            # np.savetxt("indsa2.txt", indsa2, fmt="%d")
            assert indsa.nonzero().size()[0] == size, ('Not good number of bins')
            zeros = torch.zeros((1, indsa.size()[1]), device=inds.device).byte()
            # if self.cuda:
            #     zeros = zeros.cuda()
            indsb = torch.cat((zeros, indsa))[:self.tsize, :]
            # np.savetxt("indsb.txt", indsb.cpu().numpy(), fmt="%d")
            s_repeat_[~(indsb | indsa)] = 0
            s_repeat_[indsa] = (s_repeat_ - Variable(self.t))[indsa] / self.step
            s_repeat_[indsb] = (-s_repeat_ + Variable(self.t))[indsb] / self.step

            return s_repeat_.sum(1) / size

        labels = data_batch["cls_labels"]
        self.t = self.t.to(labels.device)

        batch_size = labels.size(0)
        target_instance_num = batch_size - self.support_instance_num
        support_labels = labels[:self.support_instance_num]
        target_labels = labels[self.support_instance_num:]

        direct_loss = F.cross_entropy(preds['direct_cls_logits'], target_labels) * self.direct_pred_weight

        support_labels_ext = support_labels.unsqueeze(0).expand(target_instance_num, self.support_instance_num)
        target_labels_ext = target_labels.unsqueeze(1).expand(target_instance_num, self.support_instance_num)

        mask_pos = torch.eq(support_labels_ext, target_labels_ext).byte()
        mask_neg = ~mask_pos

        pos_num = mask_pos.sum().item()
        neg_num = mask_neg.sum().item()

        pos_inds = mask_pos.view(-1).repeat(self.tsize, 1)
        neg_inds = mask_neg.view(-1).repeat(self.tsize, 1)

        s = preds['inner_product'].view(1, -1)
        s_repeat = s.repeat(self.tsize, 1)
        s_repeat_floor = (torch.floor(s_repeat.data / self.step) * self.step).float()

        histogram_pos = histogram(pos_inds, pos_num)
        # import numpy as np
        # np.savetxt("hist_pos.txt", histogram_pos.detach().cpu().numpy(), fmt="%.4f")

        assert_almost_equal(histogram_pos.sum().item(), 1, decimal=2,
                            err_msg='Not good positive histogram', verbose=True)
        histogram_neg = histogram(neg_inds, neg_num)
        # np.savetxt("hist_neg.txt", histogram_neg.detach().cpu().numpy(), fmt="%.4f")

        assert_almost_equal(histogram_neg.sum().item(), 1, decimal=2,
                            err_msg='Not good negative histogram', verbose=True)
        histogram_pos_repeat = histogram_pos.view(-1, 1).repeat(1, histogram_pos.size()[0])
        histogram_pos_inds = torch.tril(torch.ones(histogram_pos_repeat.size()), -1).byte()

        histogram_pos_repeat[histogram_pos_inds] = 0
        histogram_pos_cdf = histogram_pos_repeat.sum(0)
        histo_loss = torch.sum(histogram_neg * histogram_pos_cdf)

        loss_dict = {
            'histo_loss': histo_loss,
            'direct_loss': direct_loss,
        }

        return loss_dict


class HistoNetMetric(nn.Module):
    def __init__(self, class_num_per_batch, batch_support_num_per_class):
        super(HistoNetMetric, self).__init__()
        self.class_num_per_batch = class_num_per_batch
        self.batch_support_num_per_class = batch_support_num_per_class
        self.support_instance_num = class_num_per_batch * batch_support_num_per_class

    def forward(self, preds, data_batch):
        labels = data_batch["cls_labels"]
        target_labels = labels[self.support_instance_num:]

        acc = preds['inner_product_preds'].eq(target_labels).float()

        return {"acc": acc}


def build_model(cfg):
    net = NetworkWrapper(cfg)

    loss_fn = HistoNetLoss(
        num_steps=cfg.HISTO_NET.NUM_STEPS,
        direct_pred_weight=cfg.HISTO_NET.DIRECT_PRED_WEIGHT,
        class_num_per_batch=cfg.DATASET.HISTO.CLASS_NUM_PER_BATCH,
        batch_support_num_per_class=cfg.DATASET.HISTO.BATCH_SUPPORT_NUM_PER_CLASS,
    )

    metric_fn = HistoNetMetric(
        class_num_per_batch=cfg.DATASET.HISTO.CLASS_NUM_PER_BATCH,
        batch_support_num_per_class=cfg.DATASET.HISTO.BATCH_SUPPORT_NUM_PER_CLASS,
    )

    return net, loss_fn, metric_fn


if __name__ == "__main__":
    from shaper_histo.config import cfg

    cfg.MODEL.TYPE = "POINTNET"
    cfg.DATASET.NUM_CLASSES = 2
    cfg.DATASET.HISTO.BATCH_SUPPORT_NUM_PER_CLASS = 1
    cfg.freeze()

    net, loss_fn, metric_fn = build_model(cfg)
    net.cuda()

    batch_size = cfg.DATASET.HISTO.CLASS_NUM_PER_BATCH * cfg.DATASET.HISTO.BATCH_SUPPORT_NUM_PER_CLASS \
                 + cfg.DATASET.HISTO.TRAIN_BATCH_TARGET_NUM
    num_points = 1024
    in_channels = 3

    data = torch.randn(batch_size, in_channels, num_points).cuda()
    cls_labels = torch.randint(cfg.DATASET.NUM_CLASSES, size=(batch_size,)).cuda()
    data_batch = {"points": data, "cls_labels": cls_labels}

    out_dict = net(data_batch)
    for k, v in out_dict.items():
        print('Histo Net:', k, v.shape)

    loss = loss_fn(out_dict, data_batch)
    for k, v in loss.items():
        print('Histo Loss {}: {}'.format(k, v))
