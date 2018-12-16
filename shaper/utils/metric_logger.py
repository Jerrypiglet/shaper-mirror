# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict
from collections import deque

import numpy as np
import torch


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
        return getattr(self, attr)

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


class AverageMeterV2(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average. Support non-scalar values.
    """

    def __init__(self, window_size=20):
        self.values = deque(maxlen=window_size)
        self.counts = deque(maxlen=window_size)
        self.sum = None
        self.count = None

    def update(self, value, count=1):
        self.values.append(value)
        self.counts.append(count)
        if self.sum is None:
            self.sum = value
        else:
            self.sum += value
        if self.count is None:
            self.count = count
        else:
            self.count += count

    @property
    def avg(self):
        avg_all = np.sum(self.values, axis=0) / np.maximum(np.sum(self.counts, axis=0), 1.0)
        return np.mean(avg_all)

    @property
    def global_avg(self):
        return np.mean(self.sum / np.maximum(self.count, 1.0))


class MetricLoggerV2(MetricLogger):
    """Support non-scalar metrics"""

    def __init__(self, delimiter="\t"):
        self.meters = dict()
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, arg in kwargs.items():
            if isinstance(arg, tuple):
                if k not in self.meters:
                    self.meters[k] = AverageMeterV2()
                value, count = arg
                value = value.cpu().numpy()
                count = count.cpu().numpy()
            else:
                if k not in self.meters:
                    self.meters[k] = AverageMeter()
                if isinstance(arg, torch.Tensor):
                    if arg.numel() == 1:
                        value = arg.item()
                        count = 1
                    else:
                        value = arg.sum().item()
                        count = arg.numel()
                    if k not in self.meters:
                        self.meters[k] = AverageMeter()
                    self.meters[k].update(value, count)
                else:
                    assert isinstance(arg, (float, int))
                    value = arg
                    count = 1

            self.meters[k].update(value, count)
