import time
import os.path as osp

from .metric_logger import AverageMeter
from tensorboardX import SummaryWriter
from .io import mkdir

_KEYWORDS = ("loss", "acc")


class TensorboardLogger(object):
    def __init__(self, log_dir, keywords=_KEYWORDS):
        self.log_dir = osp.join(log_dir, "events.{}".format(time.strftime("%m_%d_%H_%M_%S")))
        mkdir(self.log_dir)
        self.keywords = keywords
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def add_scalars(self, meters, step, prefix=""):
        for k, meter in meters.items():
            for keyword in _KEYWORDS:
                if keyword in k:
                    if isinstance(meter, AverageMeter):
                        v = meter.global_avg
                    elif isinstance(meter, (int, float)):
                        v = meter
                    else:
                        raise TypeError()

                    self.writer.add_scalar(osp.join(prefix, k), v, global_step=step)
