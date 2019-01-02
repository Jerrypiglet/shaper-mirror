# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch
from shaper.utils.checkpoint import Checkpointer as DirCheckpointer
from shaper_few_shot.utils.get_md5 import get_md5_for_file
from shaper.nn.freeze_weight import freeze_by_patterns


class Checkpointer(DirCheckpointer):
    def load(self, f=None, resume=True, load_pretrain=False, freeze_params=()):
        resume_success = False
        load_pretrain_file = load_pretrain
        if resume and self.has_checkpoint():
            # if there is existing checkpoint in path, do not load pretrain weight
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
            load_pretrain_file = False
            resume_success = True
        if not f:
            # no checkpoint could be found
            assert (not load_pretrain_file)
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}, resume_success
        self.logger.info("Loading checkpoint from {}, MD5: {}".format(f, get_md5_for_file(f)))

        if load_pretrain_file:
            checkpoint = self._load_file(f)
            pretrain_weight = checkpoint.pop("model")
            pretrain_weight = {k: v for k, v in pretrain_weight.items() if "classifier" not in k}
            model_dict = self.model.state_dict()
            model_dict.update(pretrain_weight)
            self.model.load_state_dict(model_dict)
            checkpoint.pop("optimizer")
            checkpoint.pop("scheduler")
        else:
            checkpoint = self._load_file(f)
            self.model.load_state_dict(checkpoint.pop("model"))

            if "optimizer" in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(f))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))

            if "scheduler" in checkpoint and self.scheduler:
                self.logger.info("Loading scheduler from {}".format(f))
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        if load_pretrain:
            freeze_by_patterns(self.model, freeze_params)

            # if freeze:
            #     for name, params in self.model.named_parameters():
            #         if "classifier" not in name:
            #             params.requires_grad = False

        # return any further checkpoint data
        return checkpoint, resume_success
