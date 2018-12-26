import logging
import time

import torch
from torch import nn

from shaper_fewshot.models import build_model
from shaper.solver import build_optimizer
from shaper.solver.lr_scheduler import WarmupMultiStepLR
from shaper_fewshot.data import build_dataloader
from shaper_fewshot.utils.checkpoint import CheckpointerFewshot
from shaper.utils.tensorboard_logger import TensorboardLogger
from shaper.engine.trainer import train_model, validate_model
from shaper.nn.freeze_weight import freeze_params, freeze_modules, check_frozen_params, check_frozen_modules


def train(cfg, output_dir=""):
    logger = logging.getLogger("shaper.trainer")

    # build model
    model, loss_fn, metric_fn = build_model(cfg)
    logger.info("Build model:\n{}".format(str(model)))
    model = nn.DataParallel(model).cuda()

    # build optimizer
    optimizer = build_optimizer(cfg, model)

    if cfg.SOLVER.WARMUP_STEP > 0:
        scheduler = WarmupMultiStepLR(optimizer,
                                      milestones=cfg.SOLVER.STEPS,
                                      warmup_step=cfg.SOLVER.WARMUP_STEP,
                                      warmup_gamma=cfg.SOLVER.WARMUP_GAMMA,
                                      gamma=cfg.SOLVER.GAMMA,
                                      )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=cfg.SOLVER.STEPS,
                                                         gamma=cfg.SOLVER.GAMMA)

    # build checkpointer
    checkpointer = CheckpointerFewshot(model,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       save_dir=output_dir,
                                       logger=logger)

    checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, resume=cfg.AUTO_RESUME, pretrained=True)

    # freeze modules and params
    freeze_modules(model, cfg.TRAIN.FROZEN_MODULES)
    freeze_params(model, cfg.TRAIN.FROZEN_PARAMS)
    check_frozen_modules(model, logger)
    check_frozen_params(model, logger)

    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build data loader
    train_data_loader = build_dataloader(cfg, mode="train")
    logger.info("Training data md5: {}".format(train_data_loader.dataset.get_md5_list()))

    val_period = cfg.TRAIN.VAL_PERIOD
    val_data_loader = build_dataloader(cfg, mode="val") if val_period > 0 else None
    if val_data_loader is not None:
        logger.info("Validation data md5: {}".format(val_data_loader.dataset.get_md5_list()))

    # build tensorboard logger (optionally by comment)
    tensorboard_logger = TensorboardLogger(output_dir)

    # train
    best_metric_name = "best_{}".format(cfg.TRAIN.VAL_METRIC)
    max_epoch = cfg.SOLVER.MAX_EPOCH
    start_epoch = checkpoint_data.get("epoch", 0)
    best_metric = checkpoint_data.get(best_metric_name, None)
    logger.info("Start training from epoch {}".format(start_epoch))
    for epoch in range(start_epoch, max_epoch):
        cur_epoch = epoch + 1
        scheduler.step()
        start_time = time.time()

        train_meters = train_model(model,
                                   loss_fn,
                                   metric_fn,
                                   train_data_loader,
                                   optimizer=optimizer,
                                   log_period=cfg.TRAIN.LOG_PERIOD,
                                   )
        epoch_time = time.time() - start_time
        logger.info("Epoch[{}]-Train {}  total_time: {:.2f}s".format(
            cur_epoch, train_meters.summary_str, epoch_time))

        tensorboard_logger.add_scalars(train_meters.meters, cur_epoch, prefix="train")

        # checkpoint
        if cur_epoch % ckpt_period == 0 or cur_epoch == max_epoch:
            checkpoint_data["epoch"] = cur_epoch
            checkpoint_data[best_metric_name] = best_metric
            checkpointer.save("model_{:03d}".format(cur_epoch), **checkpoint_data)

        # validate
        if val_period < 1:
            continue
        if cur_epoch % val_period == 0 or cur_epoch == max_epoch:
            val_meters = validate_model(model,
                                        loss_fn,
                                        metric_fn,
                                        val_data_loader,
                                        log_period=cfg.TEST.LOG_PERIOD,
                                        )
            logger.info("Epoch[{}]-Val {}".format(cur_epoch, val_meters.summary_str))

            tensorboard_logger.add_scalars(val_meters.meters, cur_epoch, prefix="val")

            # best validation
            cur_metric = val_meters.meters[cfg.TRAIN.VAL_METRIC].global_avg
            if best_metric is None or cur_metric > best_metric:
                best_metric = cur_metric
                checkpoint_data["epoch"] = cur_epoch
                checkpoint_data[best_metric_name] = best_metric
                checkpointer.save("model_best", **checkpoint_data)

    logger.info("Best val-{} = {}".format(cfg.TRAIN.VAL_METRIC, best_metric))

    return model
