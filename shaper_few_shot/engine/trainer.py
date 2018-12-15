import logging
import time

import torch
from torch import nn

from shaper_few_shot.models import build_model
from shaper.solver import build_optimizer
from shaper_few_shot.data import build_dataloader
from shaper.utils.torch_util import set_random_seed
from shaper_few_shot.utils.checkpoint import Checkpointer
from shaper.utils.metric_logger import MetricLogger
from shaper.utils.tensorboard_logger import TensorboardLogger
from shaper.nn.freeze_weight import unfreeze_by_patterns, _unfreeze_all_params


def train_model(model,
                loss_fn,
                metric_fn,
                data_loader,
                optimizer,
                log_period=1):
    logger = logging.getLogger("shaper.train")
    meters = MetricLogger(delimiter="  ")
    model.train()
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

    # build model
    model, loss_fn, metric_fn = build_model(cfg)
    logger.info("Build model:\n{}".format(str(model)))
    model = nn.DataParallel(model).cuda()

    # build optimizer
    optimizer = build_optimizer(cfg, model)

    # TODO: build lr scheduler
    if cfg.TRAIN.WARM_UP.ENABLE:
        def epoch_func(epoch):
            warmup_step = len(cfg.TRAIN.WARM_UP.WARM_STEP_LAMBDA)
            if epoch < warmup_step:
                return cfg.TRAIN.WARM_UP.WARM_STEP_LAMBDA[epoch]
            else:
                return cfg.TRAIN.WARM_UP.GAMMA ** ((epoch - warmup_step) // cfg.TRAIN.WARM_UP.STEP_SIZE)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, epoch_func)

    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=cfg.SOLVER.STEPS,
                                                         gamma=cfg.SOLVER.GAMMA)

    # build checkpointer
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                logger=logger)

    # for name, param in model.named_parameters():
    #     print(name, param)

    checkpoint_data, resume_success = checkpointer.load(cfg.MODEL.WEIGHT, resume=cfg.AUTO_RESUME,
                                                        load_pretrain=cfg.TRAIN.LOAD_PRETRAIN,
                                                        freeze_params=cfg.TRAIN.FREEZE_PARAMS)

    # for name, param in model.named_parameters():
    #     print(name, param)

    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build data loader
    train_data_loader = build_dataloader(cfg, mode="train")
    logger.info("Train Data MD5: {}".format(train_data_loader.dataset.get_md5_list()))

    val_period = cfg.TRAIN.VAL_PERIOD
    val_data_loader = build_dataloader(cfg, mode="val") if val_period > 0 else None
    if val_data_loader:
        logger.info("Validation Data Md5: {}".format(val_data_loader.dataset.get_md5_list()))

    # build tensorboard logger (optionally by comment)
    tensorboard_logger = TensorboardLogger(output_dir)

    # train
    best_metric_name = "best_{}".format(cfg.TRAIN.VAL_METRIC)
    max_epoch = cfg.SOLVER.MAX_EPOCH
    if "FewShot" in cfg.DATASET.TYPE and not resume_success:
        start_epoch = 0
        best_metric = None
    else:
        start_epoch = checkpoint_data.get("epoch", 0)
        best_metric = checkpoint_data.get(best_metric_name, None)
    logger.info("Start training from epoch {}".format(start_epoch))
    for epoch in range(start_epoch, max_epoch):
        cur_epoch = epoch + 1
        scheduler.step()
        start_time = time.time()
        if cfg.TRAIN.MID_UNFREEZE.ENABLE and epoch > cfg.TRAIN.MID_UNFREEZE.STEPS:
            if not cfg.TRAIN.MID_UNFREEZE.PATTERNS:
                _unfreeze_all_params(model)
            else:
                unfreeze_by_patterns(cfg.TRAIN.MID_UNFREEZE.PATTERNS)

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
            # fo = open("trials/epoch_{}_freeze_{}.txt".format(cur_epoch, cfg.TRAIN.FREEZE_PARAMS), "w")
            # for name, params in model.named_parameters():
            #     fo.write(name+":\n")
            #     fo.write(str(params))
            #     fo.write("\n")
            #     break
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
