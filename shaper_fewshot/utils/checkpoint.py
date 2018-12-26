from shaper.utils.checkpoint import Checkpointer
from .md5 import get_file_md5


class CheckpointerFewshot(Checkpointer):
    def load(self, f=None, resume=True, pretrained=False):
        if resume and self.has_checkpoint():
            # If there is existing checkpoint in path, do not load pretrain weight.
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
            pretrained = False
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}, MD5: {}".format(f, get_file_md5(f)))

        if pretrained:
            # If pretrained weight is used, abort all checkpoint data.
            checkpoint = self._load_file(f)
            pretrained_weight = checkpoint.pop("model")
            pretrained_weight = {k: v for k, v in pretrained_weight.items() if "classifier" not in k}
            model_dict = self.model.state_dict()
            model_dict.update(pretrained_weight)
            self.model.load_state_dict(model_dict)
            checkpoint.pop("optimizer")
            checkpoint.pop("scheduler")
            checkpoint = {}
        else:
            checkpoint = self._load_file(f)
            self.model.load_state_dict(checkpoint.pop("model"))
            if "optimizer" in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(f))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            if "scheduler" in checkpoint and self.scheduler:
                self.logger.info("Loading scheduler from {}".format(f))
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint
