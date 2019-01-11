from __future__ import division
from bisect import bisect_right

import torch.optim.lr_scheduler as LR


class WarmupMultiStepLR(LR._LRScheduler):
    def __init__(self, optimizer, milestones, warmup_step, gamma=0.1, warmup_gamma=0.1, last_epoch=-1):
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_step = warmup_step
        self.warmup_gamma = warmup_gamma
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_step:
            return [base_lr * (self.warmup_gamma + self.last_epoch / self.warmup_step * (1 - self.warmup_gamma))
                    for base_lr in self.base_lrs]
        else:
            return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch - self.warmup_step)
                    for base_lr in self.base_lrs]
