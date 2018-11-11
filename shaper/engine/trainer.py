import logging
import time

import torch
from torch import nn
import torch.nn.functional as F

from shaper.models import build_model
from shaper.solver import build_optimizer
from shaper.data import build_dataloader
from shaper.utils.torch_utils import set_random_seed


def train_model(model,
                loss_fn,
                data_loader,
                optimizer,):
    logger = logging.getLogger("shaper.trainer")
    model.train()
    end = time.time()
    for iteration, data_batch in enumerate(data_loader):
        data_time = time.time() - end

        points, targets = data_batch

        points = points.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        data_batch = {
            "points": points,
            "cls_labels": targets,
        }

        preds = model(points)

        optimizer.zero_grad()
        loss_dict = loss_fn(preds, data_batch)
        total_loss = sum(loss_dict.values())
        total_loss.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()

        with torch.no_grad():
            cls_logits = preds["cls_logits"]
            pred_labels = cls_logits.argmax(1)
            acc = pred_labels.eq(targets).float().mean()
            logger.info('Iter{}: loss={:.3f}, acc={:.2f}%, data={:.2f}s, time={:.2f}s'.format(
                iteration, total_loss.item(), 100 * acc.item(), data_time, batch_time))


def train(cfg):
    set_random_seed(cfg.RNG_SEED)
    logger = logging.getLogger("shaper.trainer")

    # build model
    model, loss_fn = build_model(cfg)
    device_ids = cfg.DEVICE_IDS if cfg.DEVICE_IDS else None
    model = nn.DataParallel(model, device_ids=device_ids).cuda()

    # build optimizer
    optimizer = build_optimizer(cfg, model)

    # TODO: build lr scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=cfg.SOLVER.STEPS,
                                                     gamma=cfg.SOLVER.GAMMA)

    # TODO: build checkpointer

    # TODO: build data loader
    train_data_loader = build_dataloader(cfg, is_train=True)

    # train
    logger.info("Start training")
    for epoch in range(cfg.SOLVER.MAX_EPOCH):
        scheduler.step()
        train_model(model,
                    loss_fn,
                    train_data_loader,
                    optimizer=optimizer,
                    )

    return model
