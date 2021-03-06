"""Generalized trainer"""

import logging
import time

import torch
from torch import nn

from shaper.models.build import build_model
from shaper.solver import build_optimizer, build_scheduler
from shaper.nn.freezer import Freezer
from shaper.data import build_dataloader
from shaper.utils.checkpoint import Checkpointer
from shaper.utils.metric_logger import MetricLogger
from shaper.utils.tensorboard_logger import TensorboardLogger
from shaper.utils.torch_util import set_random_seed


def train_model(model,
                loss_fn,
                metric_fn,
                data_loader,
                optimizer,
                freezer=None,
                log_period=1):
    logger = logging.getLogger("shaper.train")
    meters = MetricLogger(delimiter="  ")
    model.train()
    if freezer is not None:
        freezer.freeze()
    loss_fn.train()
    metric_fn.train()

    end = time.time()
    for iteration, data_batch in enumerate(data_loader):
        data_time = time.time() - end

        data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

        preds = model(data_batch)

        optimizer.zero_grad()
        loss_dict = loss_fn(preds, data_batch)
        metric_dict = metric_fn(preds, data_batch)
        losses = sum(loss_dict.values())
        meters.update(loss=losses, **loss_dict, **metric_dict)
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        if iteration % log_period == 0:
            logger.info(
                meters.delimiter.join(
                    [
                        "iter: {iter:4d}",
                        "{meters}",
                        "lr: {lr:.2e}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )
    return meters


def validate_model(model,
                   loss_fn,
                   metric_fn,
                   data_loader,
                   log_period=1):
    logger = logging.getLogger("shaper.validate")
    meters = MetricLogger(delimiter="  ")
    model.eval()
    loss_fn.eval()
    metric_fn.eval()

    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(data_loader):
            data_time = time.time() - end

            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

            preds = model(data_batch)

            loss_dict = loss_fn(preds, data_batch)
            metric_dict = metric_fn(preds, data_batch)
            losses = sum(loss_dict.values())
            meters.update(loss=losses, **loss_dict, **metric_dict)
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            if iteration % log_period == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "iter: {iter:4d}",
                            "{meters}",
                        ]
                    ).format(
                        iter=iteration,
                        meters=str(meters),
                    )
                )
    return meters


def train(cfg, output_dir=""):
    logger = logging.getLogger("shaper.trainer")

    set_random_seed(cfg.RNG_SEED)
    # Build model
    model, loss_fn, metric_fn = build_model(cfg)
    logger.info("Build model:\n{}".format(str(model)))
    model = nn.DataParallel(model).cuda()

    # Build optimizer
    optimizer = build_optimizer(cfg, model)

    # Build lr scheduler
    scheduler = build_scheduler(cfg, optimizer)

    # Build checkpointer
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir)

    checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, resume=cfg.AUTO_RESUME)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # Build freezer
    if cfg.TRAIN.FROZEN_PATTERNS:
        freezer = Freezer(model, cfg.TRAIN.FROZEN_PATTERNS)
        freezer.freeze(verbose=True)  # sanity check
    else:
        freezer = None

    set_random_seed(cfg.RNG_SEED)
    # Build data loader
    train_data_loader = build_dataloader(cfg, mode="train")
    val_period = cfg.TRAIN.VAL_PERIOD
    val_data_loader = build_dataloader(cfg, mode="val") if val_period > 0 else None

    # Build tensorboard logger (optionally by comment)
    tensorboard_logger = TensorboardLogger(output_dir)

    # ---------------------------------------------------------------------------- #
    # Epoch-based training
    # ---------------------------------------------------------------------------- #
    max_epoch = cfg.SCHEDULER.MAX_EPOCH
    start_epoch = checkpoint_data.get("epoch", 0)
    best_metric_name = "best_{}".format(cfg.TRAIN.VAL_METRIC)
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
                                   freezer=freezer,
                                   log_period=cfg.TRAIN.LOG_PERIOD,
                                   )
        epoch_time = time.time() - start_time
        logger.info("Epoch[{}]-Train {}  total_time: {:.2f}s".format(
            cur_epoch, train_meters.summary_str, epoch_time))

        tensorboard_logger.add_scalars(train_meters.meters, cur_epoch, prefix="train")

        # Checkpoint
        if cur_epoch % ckpt_period == 0 or cur_epoch == max_epoch:
            checkpoint_data["epoch"] = cur_epoch
            checkpoint_data[best_metric_name] = best_metric
            checkpointer.save("model_{:03d}".format(cur_epoch), **checkpoint_data)

        # Validate
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
            cur_metric = val_meters.meters.get(cfg.TRAIN.VAL_METRIC)
            # Do not save best-val epoch if no val_metric is given.
            if cur_metric is not None:
                cur_metric = cur_metric.global_avg
                if best_metric is None or cur_metric > best_metric:
                    best_metric = cur_metric
                    checkpoint_data["epoch"] = cur_epoch
                    checkpoint_data[best_metric_name] = best_metric
                    checkpointer.save("model_best", **checkpoint_data)

    logger.info("Best val-{} = {}".format(cfg.TRAIN.VAL_METRIC, best_metric))

    return model
