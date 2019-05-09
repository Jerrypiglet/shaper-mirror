"""Generalized trainer"""

import logging
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical,Normal

from shaper.models.build import build_model
from shaper.solver import build_optimizer, build_scheduler
from shaper.nn.freezer import Freezer
from shaper.data import build_dataloader
from shaper.data.datasets.utils import normalize_batch_points
from shaper.utils.checkpoint import Checkpointer
from shaper.utils.metric_logger import MetricLogger
from shaper.utils.tensorboard_logger import TensorboardLogger
from shaper.utils.torch_util import set_random_seed
from shaper.nn.functional import pdist


def train_model(models,
                loss_fns,
                data_loader,
                optimizers,
                num_zoom_iteration=1,
                meta_data_size=32,
                freezers=None,
                log_period=1,
                epoch=0):
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

        viewed_mask = torch.zeros(batch_size,1,num_point).cuda()
        predict_mask = torch.zeros(batch_size, 1,num_point).cuda()
        ones = torch.ones((batch_size, num_point)).cuda()
        tarange = torch.arange(num_point, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1).cuda()#contiguous().cuda(non_blocking=True)

        for optimizer in optimizers:
            optimizer.zero_grad()


        #for zoom_iteration in range(num_zoom_iteration):
        zoom_iteration=-1
        continue_flag=True
        while True:
            zoom_iteration+=1
            data_batch['points_and_masks'] = torch.cat([points, (viewed_mask>0).float(),(predict_mask/(viewed_mask+1e-12)).float()], 1)
            #data_batch['points_and_masks'] = torch.cat([points, meta_mask], 1)
            data_batch['viewed_mask'] = viewed_mask

            proposal_preds = proposal_model(data_batch, 'points_and_masks')




            proposal_mask = proposal_preds['mask_output'][:,0,:]

            proposal_mask = F.softmax(proposal_mask,1)
            radius_mask = proposal_preds['mask_output'][:,1,:]
            #meta_data = proposal_preds['mask_output'][:,1:,:] #B x M x N
            m,_ = torch.max(proposal_mask, 1, keepdim=True)
            proposal_mask[proposal_mask < 0.1*m]=0
            proposal_mask/=(torch.sum(proposal_mask,1, keepdim=True))
            distr = Categorical(proposal_mask)
            #distr = Categorical (torch.tensor([0.25,0.25,0.5]))
            centroids = distr.sample()
            centroids = centroids.view(batch_size,1, 1)

            gathered_centroids = points.gather(2,centroids.expand(batch_size, 3, 1))
            gathered_radius = radius_mask.gather(1,centroids.squeeze(-1))

            dists = (full_points - gathered_centroids)**2
            dists = torch.sum(dists, 1)
            nearest_dists, nearest_indices = torch.topk(dists, num_point, 1, largest=False, sorted=False)



            for b in range(batch_size):
                crop_size = Normal(1, 0.8).sample()
                #crop_size = max(0, crop_size)
                crop_size = 2**crop_size
                nearest_indices_temp = torch.nonzero(dists[b] < gathered_radius[b]*crop_size)
                if nearest_indices_temp.shape[0] >= num_point:
                    nearest_indices[b] = nearest_indices_temp[:num_point,0]



            #crop_size = Normal(1, 0.5).sample()
            #crop_size = max(0, crop_size)
            #crop_size = 2**crop_size * num_point
            #crop_size = int(crop_size)
            #nearest_dists, nearest_indices = torch.topk(dists, crop_size, 1, largest=False, sorted=False)
            #zoomed_points = full_points.gather(2, nearest_indices.view(batch_size, 1, crop_size).expand(batch_size, 3, crop_size))
            zoomed_points = full_points.gather(2, nearest_indices.view(batch_size, 1, num_point).expand(batch_size, 3, num_point))


            zoomed_points = zoomed_points.transpose_(2,1)
            zoomed_points = normalize_batch_points(zoomed_points)

            if data_loader.dataset.transform is not None:
                for b in range(batch_size):
                    zoomed_points[b] = data_loader.dataset.transform(zoomed_points[b])

            zoomed_points = zoomed_points.transpose_(2,1)
            zoomed_points = zoomed_points[:,:, :num_point]


            #zoomed_ins_seg_label = full_ins_seg_label.gather(2, nearest_indices.view(batch_size, 1, crop_size).expand(batch_size, num_ins_mask, crop_size))
            zoomed_ins_seg_label = full_ins_seg_label.gather(2, nearest_indices.view(batch_size, 1, num_point).expand(batch_size, num_ins_mask, num_point))
            zoomed_ins_seg_label = zoomed_ins_seg_label[:,:,:num_point]
            point2group = data_batch['point2group']
            point2group = torch.cat([tarange, point2group],1)
            point2group = point2group.type(torch.long)
            #groups = point2group.gather(1, nearest_indices.view(batch_size,  crop_size))
            groups = point2group.gather(1, nearest_indices.view(batch_size,  num_point))
            groups=groups[:,:num_point]
            #zoomed_meta_data = meta_data.gather(2, groups.view(batch_size, 1, num_point).expand(batch_size, meta_data_size, num_point))
            #zoomed_meta_data*=0

            #data_batch['zoomed_meta_data']=zoomed_meta_data
            data_batch['zoomed_points']=zoomed_points#torch.cat([zoomed_points,zoomed_meta_data], 1)
            #data_batch['zoomed_points']=torch.cat([zoomed_points,zoomed_meta_data], 1)
            data_batch['zoomed_ins_seg_label']=zoomed_ins_seg_label

            segmentation_preds = segmentation_model(data_batch, 'zoomed_points')
            #meta_data = segmentation_preds['mask_output'][:,-meta_data_size:,:]
            #segmentation_preds['mask_output'] = segmentation_preds['mask_output'][:,:-meta_data_size,:]

            proposal_loss_dict = proposal_loss_fn(proposal_preds, data_batch,suffix='_'+str(zoom_iteration), finish_weight = 1)
            proposal_losses = sum(proposal_loss_dict.values())
            meters.update(loss=proposal_losses, **proposal_loss_dict)
            segmentation_loss_dict = segmentation_loss_fn(segmentation_preds, data_batch, 'zoomed_ins_seg_label', suffix='_'+str(zoom_iteration))
            segmentation_losses = sum(segmentation_loss_dict.values())
            meters.update(loss=segmentation_losses, **segmentation_loss_dict)
            proposal_losses.backward(retain_graph=True)
            segmentation_losses.backward(retain_graph = continue_flag)

            masks = segmentation_preds['mask_output']
            masks = F.softmax(masks,1)
            confs = segmentation_preds['global_output']
            confs = torch.sigmoid(confs)
            masks *= confs.unsqueeze(-1)

            masks, _ = torch.max(masks, 1)

            predict_mask = predict_mask.squeeze(1)
            viewed_mask=viewed_mask.squeeze(1)
            new_groups = groups.unsqueeze(1).expand(batch_size, meta_data_size, num_point).detach()
            predict_mask = predict_mask.scatter_add(1,groups, masks)
            viewed_mask = viewed_mask.scatter_add(1,groups, ones)
            #meta_mask = meta_mask.scatter(2,new_groups, meta_data)
            #viewed_mask[viewed_mask>=1]=1
            viewed_mask = viewed_mask.unsqueeze(1).detach()
            predict_mask = predict_mask.unsqueeze(1).detach()

            if not continue_flag:
                break

            continue_flag = torch.sum(data_batch['finish_label'].detach()).cpu() > 0


            if zoom_iteration % 10==9:
                for optimizer in optimizers:
                    optimizer.step()

        for optimizer in optimizers:
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
                        #"lr: {lr:.2e}",
                        #"max mem: {memory:.0f}",
                    ]
                ).format(
                    iter=iteration,
                    meters=str(meters),
                    #lr=optimizers[0].param_groups[0]["lr"],
                    #memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )
    return meters


def validate_model(models,
                   loss_fns,
                   data_loader,
                   num_zoom_iteration=1,
                   log_period=1):
    logger = logging.getLogger("shaper.validate")
    meters = MetricLogger(delimiter="  ")
    proposal_model, segmentation_model = models
    proposal_model.eval()
    segmentation_model.eval()
    proposal_loss_fn, segmentation_loss_fn = loss_fns
    proposal_loss_fn.eval()
    segmentation_loss_fn.eval()

    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(data_loader):
            data_time = time.time() - end

            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items() if type(v) is torch.Tensor}
            full_points = data_batch['full_points']
            points = data_batch['points']
            full_ins_seg_label = data_batch['full_ins_seg_label']
            num_point = points.shape[2]
            batch_size = points.shape[0]
            num_ins_mask = full_ins_seg_label.shape[1]

            viewed_mask = torch.zeros(batch_size,1,num_point).cuda()
            predict_mask = torch.zeros(batch_size, 1,num_point).cuda()
            for zoom_iteration in range(num_zoom_iteration):

                data_batch['points_and_masks'] = torch.cat([points, viewed_mask,predict_mask], 1)
                data_batch['viewed_mask'] = viewed_mask

                proposal_preds = proposal_model(data_batch, 'points_and_masks')

                proposal_mask = proposal_preds['mask_output'][:,0,:]
                proposal_mask = F.softmax(proposal_mask,1)
                #meta_data = proposal_preds['mask_output'][:,1:,:] #B x M x N
                #num_meta_data = meta_data.shape[1]
                m,_ = torch.max(proposal_mask, 1, keepdim=True)
                proposal_mask[proposal_mask < 0.1*m]=0
                proposal_mask/=(torch.sum(proposal_mask,1, keepdim=True))
                distr = Categorical(proposal_mask)
                #distr = Categorical (torch.tensor([0.25,0.25,0.5]))
                # print ('let me tell you', data_batch['point2group'].cuda())
                centroids = distr.sample()
                centroids = centroids.view(batch_size,1, 1)

                gathered_centroids = points.gather(2,centroids.expand(batch_size, 3, 1))

                dists = (full_points - gathered_centroids)**2
                dists = torch.sum(dists, 1)


                crop_size = Normal(1, 0.5).sample()
                crop_size = max(0, crop_size)
                crop_size = 2**crop_size * num_point
                crop_size = int(crop_size)
                nearest_dists, nearest_indices = torch.topk(dists, crop_size, 1, largest=False, sorted=False)
                zoomed_points = full_points.gather(2, nearest_indices.view(batch_size, 1, crop_size).expand(batch_size, 3, crop_size))


                zoomed_points = zoomed_points.transpose_(2,1)
                zoomed_points = normalize_batch_points(zoomed_points)

                if data_loader.dataset.transform is not None:
                    for b in range(batch_size):
                        zoomed_points[b] = data_loader.dataset.transform(zoomed_points[b])

                zoomed_points = zoomed_points.transpose_(2,1)
                zoomed_points = zoomed_points[:,:, :num_point]


                zoomed_ins_seg_label = full_ins_seg_label.gather(2, nearest_indices.view(batch_size, 1, crop_size).expand(batch_size, num_ins_mask, crop_size))
                zoomed_ins_seg_label = zoomed_ins_seg_label[:,:,:num_point]
                point2group = data_batch['point2group']
                point2group = torch.cat([torch.arange(num_point, dtype=torch.int32).view(1,num_point).expand(batch_size, num_point).contiguous().cuda(non_blocking=True), point2group],1)
                point2group = point2group.type(torch.long)
                groups = point2group.gather(1, nearest_indices.view(batch_size,  crop_size))
                groups=groups[:,:num_point]
                #zoomed_meta_data = meta_data.gather(2, groups.view(batch_size, 1, num_point).expand(batch_size, num_meta_data, num_point))

                #data_batch['zoomed_meta_data']=zoomed_meta_data
                data_batch['zoomed_points']=zoomed_points#torch.cat([zoomed_points,zoomed_meta_data], 1)
                data_batch['zoomed_ins_seg_label']=zoomed_ins_seg_label

                segmentation_preds = segmentation_model(data_batch, 'zoomed_points')

                proposal_loss_dict = proposal_loss_fn(proposal_preds, data_batch)
                proposal_losses = sum(proposal_loss_dict.values())
                meters.update(loss=proposal_losses, **proposal_loss_dict)
                segmentation_loss_dict = segmentation_loss_fn(segmentation_preds, data_batch, 'zoomed_ins_seg_label')
                segmentation_losses = sum(segmentation_loss_dict.values())
                meters.update(loss=segmentation_losses, **segmentation_loss_dict)
                exit(0)



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
    # Build data loader
    train_data_loader = build_dataloader(cfg, mode="train")
    val_period = cfg.TRAIN.VAL_PERIOD
    val_data_loader = build_dataloader(cfg, mode="val") if val_period > 0 else None

    num_gt_masks = int(train_data_loader.dataset.num_gt_masks*1.2+2)
    if cfg.MODEL.NUM_INS_MASKS > num_gt_masks:
        cfg.MODEL.NUM_INS_MASKS = num_gt_masks


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
        checkpointer = Checkpointer(models[i],
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    save_dir=output_dir)
        checkpointers.append(checkpointer)

        checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, tag_file='last_checkpoint_{:02d}'.format(i),resume=cfg.AUTO_RESUME)
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
                                   num_zoom_iteration=cfg.TRAIN.NUM_ZOOM_ITERATION,
                                   meta_data_size=cfg.MODEL.META_DATA,
                                   optimizers=optimizers,
                                   freezers=freezers,
                                   log_period=cfg.TRAIN.LOG_PERIOD,
                                   epoch=cur_epoch
                                   )
        epoch_time = time.time() - start_time
        logger.info("Epoch[{}]-Train {}  total_time: {:.2f}s".format(
            cur_epoch, train_meters.summary_str, epoch_time))

        tensorboard_logger.add_scalars(train_meters.meters, cur_epoch, prefix="train")

        # Checkpoint
        if cur_epoch % ckpt_period == 0 or cur_epoch == max_epoch:
            for i in range(len(checkpointers)):
                checkpoint_datas[i]["epoch"] = cur_epoch
                checkpointers[i].save("model_{:02d}_{:03d}".format(i,cur_epoch), tag_file='last_checkpoint_{:02d}'.format(i),**checkpoint_datas[i])

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

    #logger.info("Best val-{} = {}".format(cfg.TRAIN.VAL_METRIC, best_metric))

    return model
