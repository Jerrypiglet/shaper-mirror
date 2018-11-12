import logging
import time
import datetime

import torch
from torch import nn

from shaper.models import build_model
from shaper.solver import build_optimizer
from shaper.data import build_dataloader
from shaper.utils.torch_utils import set_random_seed
from shaper.utils.checkpoint import Checkpointer
from shaper.utils.metric_logger import MetricLogger


def train_model(model,
                loss_fn,
                metric_fn,
                data_loader,
                optimizer,
                log_period=1):
    logger = logging.getLogger("shaper.trainer")
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
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_cached() / 1024.0 / 1024.0,
                )
            )


def train(cfg, output_dir=""):
    set_random_seed(cfg.RNG_SEED)
    logger = logging.getLogger("shaper.trainer")

    # build model
    model, loss_fn, metric_fn = build_model(cfg)
    device_ids = cfg.DEVICE_IDS if cfg.DEVICE_IDS else None
    model = nn.DataParallel(model, device_ids=device_ids).cuda()

    # build optimizer
    optimizer = build_optimizer(cfg, model)

    # TODO: build lr scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=cfg.SOLVER.STEPS,
                                                     gamma=cfg.SOLVER.GAMMA)

    # build checkpointer
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir)

    checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT,
                                        resume=cfg.AUTO_RESUME)

    # build data loader
    train_data_loader = build_dataloader(cfg, is_train=True)

    # train
    logger.info("Start training")
    for epoch in range(checkpoint_data.get("epoch", 0), cfg.SOLVER.MAX_EPOCH):
        scheduler.step()
        logger.info("Epoch {} starts".format(epoch))
        start_time = time.time()
        train_model(model,
                    loss_fn,
                    metric_fn,
                    train_data_loader,
                    optimizer=optimizer,
                    log_period=cfg.TRAIN.LOG_PERIOD,
                    )
        epoch_time = time.time() - start_time
        eta_seconds = epoch_time * (cfg.SOLVER.MAX_EPOCH - 1 - epoch)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        logger.info("Epoch {epoch} ends within {time}s. Eta: {eta}".format(
            epoch=epoch, time=epoch_time, eta=eta_string))
        if epoch % cfg.TRAIN.CHECKPOINT_PERIOD == 0 and epoch > 0:
            checkpoint_data["epoch"] = epoch
            checkpointer.save("model_{:07d}".format(epoch),
                              **checkpoint_data)
        # TODO: add validation

    return model
