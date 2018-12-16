import os.path as osp

from .metric_logger import AverageMeter, AverageMeterV2
from tensorboardX import SummaryWriter

_KEYWORDS = ("loss", "acc", "IOU")


class TensorboardLogger(object):
    def __init__(self, log_dir, keywords=_KEYWORDS):
        self.log_dir = log_dir
        self.keywords = keywords
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalars(self, meters, step, prefix=""):
        for k, meter in meters.items():
            for keyword in _KEYWORDS:
                if keyword in k:
                    if isinstance(meter, (AverageMeter, AverageMeterV2)):
                        v = meter.global_avg
                    elif isinstance(meter, (int, float)):
                        v = meter
                    else:
                        raise TypeError()

                    self.writer.add_scalar(osp.join(prefix, k), v, global_step=step)
