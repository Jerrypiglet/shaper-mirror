"""Generalized trainer"""

import logging
import time

import torch
from torch import nn
from torch.distributions import Categorical

from shaper.models.build import build_model
from shaper.solver import build_optimizer, build_scheduler
from shaper.nn.freezer import Freezer
from shaper.data import build_dataloader
from shaper.utils.checkpoint import Checkpointer
from shaper.utils.metric_logger import MetricLogger
from shaper.utils.tensorboard_logger import TensorboardLogger
from shaper.utils.torch_util import set_random_seed


def train_model(models,
                loss_fns,
                data_loader,
                optimizers,
                freezers=None,
                log_period=1):
    logger = logging.getLogger("shaper.train")
    meters = MetricLogger(delimiter="  ")
    proposal_model, segmentation_model = models
    proposal_model.train()
    segmentation_model.train()
    for freezer in freezers:
        if freezer is not None:
            freezer.freeze()
    proposal_loss_fn, segmentation_loss_fn = loss_fns
    proposal_loss_fn.train()
    segmentation_loss_fn.train()
    #metric_fn.train()

    end = time.time()
    for iteration, data_batch in enumerate(data_loader):
        data_time = time.time() - end


        data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items() if type(v) is torch.Tensor}
        full_points = data_batch['full_points']
        points = data_batch['points']
        full_ins_seg_label = data_batch['full_ins_seg_label']
        num_point = points.shape[2]
        batch_size = points.shape[0]
        num_ins_mask = full_ins_seg_label.shape[1]

        proposal_preds = proposal_model(data_batch)

        proposal_mask = proposal_preds['mask_output'][:,0,:]
        meta_data = proposal_preds['mask_output'][:,1:,:] #B x M x N
        num_meta_data = meta_data.shape[1]
        distr = Categorical(proposal_mask)
        centroids = distr.sample()
        centroids = centroids.view(batch_size,1, 1)

        gathered_centroids = points.gather(2,centroids.expand(batch_size, 3, 1))

        dists = (full_points - gathered_centroids)**2
        dists = torch.sum(dists, 1)
        nearest_dists, nearest_indices = torch.topk(dists, num_point, 1, largest=False, sorted=False)

        zoomed_points = full_points.gather(2, nearest_indices.view(batch_size, 1, num_point).expand(batch_size, 3, num_point))
        zoomed_ins_seg_label = full_ins_seg_label.gather(2, nearest_indices.view(batch_size, 1, num_point).expand(batch_size, num_ins_mask, num_point))
        point2group = data_batch['point2group']
        point2group = torch.cat([torch.arange(num_point, dtype=torch.int32).view(1,num_point).expand(batch_size, num_point).contiguous().cuda(non_blocking=True), point2group],1)
        point2group = point2group.type(torch.long)
        groups = point2group.gather(1, nearest_indices.view(batch_size,  num_point))
        zoomed_meta_data = meta_data.gather(2, groups.view(batch_size, 1, num_point).expand(batch_size, num_meta_data, num_point))

        data_batch['zoomed_meta_data']=zoomed_meta_data
        data_batch['zoomed_points']=torch.cat([zoomed_points,zoomed_meta_data], 1)
        data_batch['zoomed_ins_seg_label']=zoomed_ins_seg_label

        segmentation_preds = segmentation_model(data_batch, 'zoomed_points')


        optimizer.zero_grad()
        proposal_loss_dict = proposal_loss_fn(proposal_preds, data_batch)
        proposal_losses = sum(proposal_loss_dict.values())
        meters.update(loss=proposal_losses, **proposal_loss_dict)
        segmentation_loss_dict = segmentation_loss_fn(segmentation_preds, data_batch, 'zoomed_ins_seg_label')
        segmentation_losses = sum(segmentation_loss_dict.values())
        meters.update(loss=segmentation_losses, **segmentation_loss_dict)
        proposal_losses.backward()
        segmentation_losses.backward()
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
    #metric_fn.eval()

    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(data_loader):
            data_time = time.time() - end

            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items() if type(v) is torch.Tensor}

            preds = model(data_batch)

            loss_dict = loss_fn(preds, data_batch)
            #metric_dict = metric_fn(preds, data_batch)
            losses = sum(loss_dict.values())
            meters.update(loss=losses, **loss_dict)
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
    models, loss_fns = build_model(cfg)
    optimizers=[]
    schedulers=[]
    checkpointers=[]
    checkpoint_datas=[]
    freezers=[]
    for i in range(len(models)):
        logger.info("Build model:\n{}".format(str(models[i])))
        model = nn.DataParallel(models[i]).cuda()
        models[i]=model

        # Build optimizer
        optimizer = build_optimizer(cfg, model)
        optimizers.append(optimizer)

        # Build lr scheduler
        scheduler = build_scheduler(cfg, optimizer)
        schedulers.append(scheduler)

        # Build checkpointer
        checkpointer = Checkpointer(model,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    save_dir=output_dir)
        checkpointers.append(checkpointer)

        checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, resume=cfg.AUTO_RESUME)
        checkpoint_datas.append(checkpoint_data)

        # Build freezer
        if cfg.TRAIN.FROZEN_PATTERNS:
            freezer = Freezer(model, cfg.TRAIN.FROZEN_PATTERNS)
            freezer.freeze(verbose=True)  # sanity check
        else:
            freezer = None
        freezers.append(freezer)

    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD
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
    start_epoch = checkpoint_datas[0].get("epoch", 0)
    logger.info("Start training from epoch {}".format(start_epoch))
    for epoch in range(start_epoch, max_epoch):
        cur_epoch = epoch + 1
        for scheduler in schedulers:
            scheduler.step()
        start_time = time.time()
        train_meters = train_model(models,
                                   loss_fns,
                                   train_data_loader,
                                   optimizers=optimizers,
                                   freezers=freezers,
                                   log_period=cfg.TRAIN.LOG_PERIOD,
                                   )
        epoch_time = time.time() - start_time
        logger.info("Epoch[{}]-Train {}  total_time: {:.2f}s".format(
            cur_epoch, train_meters.summary_str, epoch_time))

        tensorboard_logger.add_scalars(train_meters.meters, cur_epoch, prefix="train")

        # Checkpoint
        if cur_epoch % ckpt_period == 0 or cur_epoch == max_epoch:
            for i in range(len(checkpointers)):
                checkpoint_datas[i]["epoch"] = cur_epoch
                checkpoint_datas[i][best_metric_name] = best_metric
                checkpointers[i].save("model_{:03d}".format(cur_epoch), **checkpoint_datas[i])

        # Validate
        if val_period < 1:
            continue
        if cur_epoch % val_period == 0 or cur_epoch == max_epoch:
            val_meters = validate_model(models,
                                        loss_fns,
                                        val_data_loader,
                                        log_period=cfg.TEST.LOG_PERIOD,
                                        )
            logger.info("Epoch[{}]-Val {}".format(cur_epoch, val_meters.summary_str))

            tensorboard_logger.add_scalars(val_meters.meters, cur_epoch, prefix="val")

    logger.info("Best val-{} = {}".format(cfg.TRAIN.VAL_METRIC, best_metric))

    return model
