#!/usr/bin/env python
"""Test point cloud part segmentation models"""

from __future__ import division
import argparse
import os.path as osp
import logging
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from shaper.config.part_instance_segmentation import cfg
from shaper.config import purge_cfg
from shaper.models.build import build_model
from shaper.data.build import build_dataloader, build_transform
from shaper.data import transforms as T
from shaper.data.datasets.evaluator import evaluate_foveal_segmentation
from shaper.utils.checkpoint import Checkpointer
from shaper.utils.metric_logger import MetricLogger
from shaper.utils.io import mkdir
from shaper.utils.logger import setup_logger
from shaper.utils.torch_util import set_random_seed



def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch 3D Deep Learning Training")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def test(cfg, output_dir=""):
    logger = logging.getLogger("shaper.tester")

    # Build model
    models, loss_fns = build_model(cfg)
    checkpointers=[]
    for i in range(len(models)):
        models[i] = nn.DataParallel(models[i]).cuda()

        # Build checkpointer
        checkpointer = Checkpointer(models[i], save_dir=output_dir)

        if cfg.TEST.WEIGHT:
            # Load weight if specified
            weight_path = cfg.TEST.WEIGHT.replace("@", output_dir)
            weight_path = weight_path.replace('#', '%02d'%i)
            checkpointer.load(weight_path,  resume=False)
        else:
            # Load last checkpoint
            checkpointer.load(None, tag_file='last_checkpoint_{:02d}'.format(i),resume=True)

    # Build data loader
    test_data_loader = build_dataloader(cfg, mode="test")
    test_dataset = test_data_loader.dataset

    # Prepare visualization
    vis_dir = cfg.TEST.VIS_DIR.replace("@", output_dir)
    if vis_dir:
        mkdir(vis_dir)

    # ---------------------------------------------------------------------------- #
    # Test
    # ---------------------------------------------------------------------------- #
    proposal_logit_all=[ [] for i in range(cfg.TEST.NUM_ZOOM_ITERATION)]
    finish_logit_all=[ [] for i in range(cfg.TEST.NUM_ZOOM_ITERATION)]
    zoomed_points_all=[ [] for i in range(cfg.TEST.NUM_ZOOM_ITERATION)]
    conf_logit_all=[ [] for i in range(cfg.TEST.NUM_ZOOM_ITERATION)]
    seg_logit_all=[ [] for i in range(cfg.TEST.NUM_ZOOM_ITERATION)]
    global_seg_logit_all=[ [] for i in range(cfg.TEST.NUM_ZOOM_ITERATION)]
    viewed_mask_all = [ [] for i in  range(cfg.TEST.NUM_ZOOM_ITERATION)]
    proposal_model, segmentation_model = models
    proposal_loss_fn, segmentation_loss_fn = loss_fns
    proposal_model.eval()
    segmentation_model.eval()
    proposal_loss_fn.eval()
    segmentation_loss_fn.eval()
    #metric_fn.eval()
    set_random_seed(cfg.RNG_SEED)

    test_meters = MetricLogger(delimiter="  ")
    with torch.no_grad():
        start_time = time.time()
        end = start_time
        for iteration, data_batch in enumerate(test_data_loader):
            data_time = time.time() - end

            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items() if type(v) == torch.Tensor}
            full_points = data_batch['full_points']
            points = data_batch['points']
            full_ins_seg_label = data_batch['full_ins_seg_label']
            num_point = points.shape[2]
            batch_size = points.shape[0]
            assert batch_size == 1
            num_ins_mask = full_ins_seg_label.shape[1]



            viewed_mask = torch.zeros(batch_size,1,num_point).cuda()
            predict_mask = torch.zeros(batch_size, 1,num_point).cuda()

            continue_flag=True
            for zoom_iteration in range(cfg.TEST.NUM_ZOOM_ITERATION):
            #zoom_iteration = - 1
            #while True:
                #zoom_iteration+=1
                data_batch['points_and_masks'] = torch.cat([points, (viewed_mask>0).float(),(predict_mask/(viewed_mask+1e-12))], 1)
                data_batch['viewed_mask'] = viewed_mask
                proposal_preds = proposal_model(data_batch, 'points_and_masks')
                ins_seg_label ,_ = torch.max(data_batch['ins_seg_label'],1)

                proposal_mask = proposal_preds['mask_output'][:,0,:]
                proposal_mask = F.softmax(proposal_mask,1)
                radius_mask = proposal_preds['mask_output'][:,1,:]
                proposal_logit_all[zoom_iteration].append(proposal_mask.cpu().numpy())
                finish_logit_all[zoom_iteration].append(torch.sigmoid(proposal_preds['global_output']).cpu().numpy())
                m,_ = torch.max(proposal_mask, 1, keepdim=True)
                proposal_mask[proposal_mask < 0.5*m]=0
                proposal_mask/=(torch.sum(proposal_mask,1, keepdim=True))
                distr = Categorical(proposal_mask)
                centroids = distr.sample()
                _, centroids = torch.max(proposal_mask,1)
                #_, centroids = torch.min(radius_mask/(proposal_mask+1e-12),1)
                centroids = centroids.view(batch_size,1, 1)

                gathered_centroids = points.gather(2,centroids.expand(batch_size, 3, 1))
                gathered_radius = radius_mask.gather(1,centroids.squeeze(-1))

                dists = (full_points - gathered_centroids)**2
                dists = torch.sum(dists, 1)

                crop_size= 2*num_point
                #nearest_dists, nearest_indices = torch.topk(dists, crop_size, 1, largest=False, sorted=False)
                nearest_dists, nearest_indices = torch.topk(dists, num_point, 1, largest=False, sorted=False)
                for b in range(batch_size):
                    nearest_indices_temp = torch.nonzero(dists[b] < gathered_radius[b]*2)
                    if nearest_indices_temp.shape[0] >= num_point:
                        nearest_indices[b] = nearest_indices_temp[:num_point,0]

                #zoomed_points = full_points.gather(2, nearest_indices.view(batch_size, 1, crop_size).expand(batch_size, 3, crop_size))
                #zoomed_points = full_points.gather(2, nearest_indices.view(batch_size, 1, crop_size).expand(batch_size, 3, crop_size))
                zoomed_points = full_points.gather(2, nearest_indices.view(batch_size, 1, num_point).expand(batch_size, 3, num_point))
                ##center zoomed points
                zoomed_points -= torch.sum(zoomed_points, 2,keepdim=True)/zoomed_points.shape[2]
                maxnorm, _ = torch.max(torch.sum(zoomed_points**2, 1),1)
                maxnorm = maxnorm ** 0.5
                maxnorm = maxnorm.view(batch_size, 1, 1)
                zoomed_points /=maxnorm

                #zoomed_ins_seg_label = full_ins_seg_label.gather(2, nearest_indices.view(batch_size, 1, crop_size).expand(batch_size, num_ins_mask, crop_size))
                zoomed_ins_seg_label = full_ins_seg_label.gather(2, nearest_indices.view(batch_size, 1, num_point).expand(batch_size, num_ins_mask, num_point))


                zoomed_points=zoomed_points[:,:,:num_point]
                zoomed_ins_seg_label = zoomed_ins_seg_label[:,:,:num_point]

                zoomed_points_all[zoom_iteration].append(zoomed_points.cpu().numpy())


                point2group = data_batch['point2group']
                point2group = torch.cat([torch.arange(num_point, dtype=torch.int32).view(1,num_point).expand(batch_size, num_point).contiguous().cuda(non_blocking=True), point2group],1)
                point2group = point2group.type(torch.long)
                #groups = point2group.gather(1, nearest_indices.view(batch_size,  crop_size))
                groups = point2group.gather(1, nearest_indices.view(batch_size,  num_point))
                groups=groups[:,:num_point]

                data_batch['zoomed_points']=zoomed_points
                data_batch['zoomed_ins_seg_label']=zoomed_ins_seg_label

                segmentation_preds = segmentation_model(data_batch, 'zoomed_points')

                seg_logit_all[zoom_iteration].append(F.softmax(segmentation_preds["mask_output"],1).cpu().numpy())
                conf_logit_all[zoom_iteration].append(torch.sigmoid(segmentation_preds["global_output"]).cpu().numpy())


                masks = segmentation_preds['mask_output']
                masks = F.softmax(masks,1)
                confs = segmentation_preds['global_output']
                confs = torch.sigmoid(confs)

                new_groups = groups.unsqueeze(1).expand(masks.shape)
                vm = torch.zeros(masks.shape).cuda()
                pm = torch.zeros(masks.shape).cuda()
                pm = pm.scatter_add(2,new_groups, masks)
                vm = vm.scatter_add(2,new_groups, torch.ones(masks.shape).cuda())
                global_seg_logit_all[zoom_iteration].append((pm/(vm+1e-12)).cpu().numpy())

                masks *= confs.unsqueeze(-1)
                masks, _ = torch.max(masks, 1)

                predict_mask = predict_mask.squeeze(1)
                viewed_mask=viewed_mask.squeeze(1)
                predict_mask = predict_mask.scatter_add(1,groups, masks)
                viewed_mask = viewed_mask.scatter_add(1,groups, torch.ones((batch_size, num_point)).cuda())
                #viewed_mask[viewed_mask>=1]=1
                viewed_mask_all[zoom_iteration].append(viewed_mask.cpu().numpy())
                viewed_mask = viewed_mask.unsqueeze(1).detach()
                predict_mask = predict_mask.unsqueeze(1).detach()




                proposal_loss_dict = proposal_loss_fn(proposal_preds, data_batch)



                #if not continue_flag:
                #    break

                continue_flag = torch.sum(data_batch['finish_label'].detach()).cpu() > 0
                #loss_dict = loss_fn(preds, data_batch)
                #metric_dict = metric_fn(preds, data_batch)
                #losses = sum(loss_dict.values())
                #test_meters.update(loss=losses, **loss_dict)

                #batch_time = time.time() - end
                #end = time.time()
                #test_meters.update(time=batch_time, data=data_time)

                #if iteration % cfg.TEST.LOG_PERIOD == 0:
                #    logger.info(
                #        test_meters.delimiter.join(
                #            [
                #                "iter: {iter:4d}",
                #                "{meters}",
                #            ]
                #        ).format(
                #            iter=iteration,
                #            meters=str(test_meters),
                #        )
                #    )
    test_time = time.time() - start_time
    logger.info("Test {}  forward time: {:.2f}s".format(test_meters.summary_str, test_time))
    for zoom_iteration in range(cfg.TEST.NUM_ZOOM_ITERATION):
        seg_logit_all[zoom_iteration] = np.concatenate(seg_logit_all[zoom_iteration], axis=0)
        global_seg_logit_all[zoom_iteration] = np.concatenate(global_seg_logit_all[zoom_iteration], axis=0)
        conf_logit_all[zoom_iteration] = np.concatenate(conf_logit_all[zoom_iteration], axis=0)
        viewed_mask_all[zoom_iteration] = np.concatenate(viewed_mask_all[zoom_iteration], axis=0)
        finish_logit_all[zoom_iteration] = np.concatenate(finish_logit_all[zoom_iteration], axis=0)
        zoomed_points_all[zoom_iteration] = np.concatenate(zoomed_points_all[zoom_iteration], axis=0)
        proposal_logit_all[zoom_iteration] = np.concatenate(proposal_logit_all[zoom_iteration], axis=0)

    return evaluate_foveal_segmentation(test_dataset, viewed_mask_all,  proposal_logit_all, finish_logit_all, zoomed_points_all, seg_logit_all, conf_logit_all,global_seg_logit_all, output_dir=output_dir, vis_dir=vis_dir)


def main():
    args = parse_args()

    # Load the configuration
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    #cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # Replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace("configs", "outputs")
        output_dir = output_dir.replace('@', config_path)
        mkdir(output_dir)

    logger = setup_logger("shaper", output_dir, prefix="test")
    logger.info("Using {} GPUs".format(torch.cuda.device_count()))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    assert cfg.TASK == "foveal_part_instance_segmentation"

    aps = np.zeros((5, 20))
    for i in range(420, 520, 20):
        print(i)
        #cfg.TEST.WEIGHT='@/model_#_%03d.pth'%i
        aps[(i-420)//20] = test(cfg, output_dir)
        print(aps[(i-420)//20])
    temp = np.mean(aps, 0, keepdims=True)
    std_dev = np.mean((aps - temp)**2,0)**0.5
    print('mean',list(zip(range(5,105,5),np.mean(aps,0))))
    print('std_dev', list(zip(range(5,105,5), std_dev)))
    aps = np.mean(aps,1)
    mean = np.mean(aps)
    std_dev = np.mean((aps - mean)**2)**0.5
    print('mean', mean)
    print('std dev', std_dev)


if __name__ == "__main__":
    main()
