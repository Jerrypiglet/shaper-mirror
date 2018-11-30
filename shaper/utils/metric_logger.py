# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict
from collections import deque

import numpy as np
import torch
import json


class AverageMeter(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.values = deque(maxlen=window_size)
        self.counts = deque(maxlen=window_size)
        self.sum = 0.0
        self.count = 0

    def update(self, value, count=1):
        self.values.append(value)
        self.counts.append(count)
        self.sum += value
        self.count += count

    @property
    def avg(self):
        return np.sum(self.values) / np.sum(self.counts)

    @property
    def global_avg(self):
        return self.sum / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            count = 1
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v = v.item()
                else:
                    count = v.numel()
                    v = v.sum().item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, count)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return object.__getattr__(self, attr)

    def __str__(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.avg, meter.global_avg)
            )
        return self.delimiter.join(metric_str)

    @property
    def summary_str(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(metric_str)


class IOULogger(object):
    def __init__(self, cfg, delimiter="\t"):
        self.intersection = torch.zeros(cfg.DATASET.NUM_CLASSES)
        self.union = torch.zeros(cfg.DATASET.NUM_CLASSES)
        self.dataset_type = cfg.DATASET.TYPE
        self.delimiter = delimiter

        if self.dataset_type == "ShapeNet":
            with open("data/shapenet/overallid_to_catid_partid.json") as f:
                data = np.array(json.load(f))
            self.ind = np.tile(np.unique(data[:, 0]), [data.shape[0], 1]) == np.expand_dims(data[:, 0], 1)
            self.cat_names = np.loadtxt("data/shapenet/all_object_categories.txt", dtype=str)[:, 0]

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert k in ["intersection", "union"]
            assert isinstance(v, torch.Tensor)
            setattr(self, k, self.__getattribute__(k) + v)

    def compute_iou(self):
        if self.dataset_type == "ShapeNet":
            per_obj_cat_intersection = (np.expand_dims(self.intersection.numpy(), 1) * self.ind).sum(axis=0)
            per_obj_cat_union = (np.expand_dims(self.union.numpy(), 1) * self.ind).sum(axis=0)
            per_obj_cat_iou = per_obj_cat_intersection / (per_obj_cat_union + 1e-10)
        else:
            raise NotImplementedError

        return per_obj_cat_iou

    def __str__(self):
        per_obj_cat_iou = self.compute_iou()

        metric_str = ["mIOU: {:.4f}".format(per_obj_cat_iou.mean())]
        for i, cat in enumerate(self.cat_names):
            metric_str.append("{}: {:.4f}".format(cat, per_obj_cat_iou[i]))

        return self.delimiter.join(metric_str)

    @property
    def summary_str(self):
        return str(self)


class AllMeters(object):
    def __init__(self, meters_list):
        self.meters_list = meters_list

    def __getattr__(self, attr):
        for meters in self.meters_list:
            if hasattr(meters, attr):
                return getattr(meters, attr)
        raise KeyError

    @property
    def summary_str(self):
        return self.meters_list[0].delimiter.join([meter.summary_str for meter in self.meters_list])
