#!/usr/bin/env python
"""Test point cloud semantic segmentation models"""

from __future__ import division
import argparse
import os.path as osp
import logging
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from shaper.config.semantic_segmentation import cfg
from shaper.config import purge_cfg
from shaper.models import build_model
from shaper.data import build_dataloader
from shaper.data import transforms as T
from shaper.data.datasets.evaluator import evaluate_semantic_segmentation
from shaper.utils.checkpoint import Checkpointer
from shaper.utils.metric_logger import MetricLogger
from shaper.utils.io import mkdir
from shaper.utils.logger import setup_logger
from shaper.utils.torch_util import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch 3D Deep Learning Testing")
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
    model, loss_fn, metric_fn = build_model(cfg)
    model = nn.DataParallel(model).cuda()

    # Build checkpointer
    checkpointer = Checkpointer(model, save_dir=output_dir)

    if cfg.TEST.WEIGHT:
        # Load weight if specified
        weight_path = cfg.TEST.WEIGHT.replace("@", output_dir)
        checkpointer.load(weight_path, resume=False)
    else:
        # Load last checkpoint
        checkpointer.load(None, resume=True)

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
    seg_logit_all = []
    model.eval()
    loss_fn.eval()
    metric_fn.eval()

    if cfg.TEST.VOTE.ENABLE:
        # Disable inherent shuffle
        test_dataset.shuffle_points = False
        # Remove old transform
        test_dataset.transform = None
        # Build new transform
        transform_list = [T.PointCloudRotateByAngle(cfg.TEST.VOTE.AXIS, 2 * np.pi * view_ind / cfg.TEST.VOTE.NUM_VIEW)
                          for view_ind in range(cfg.TEST.VOTE.NUM_VIEW)]
        if cfg.TEST.VOTE.SHUFFLE:
            # Some non-deterministic algorithms might benefit from shuffle.
            set_random_seed(cfg.RNG_SEED)

        with torch.no_grad():
            start_time = time.time()
            end = start_time
            for ind in range(len(test_dataset)):
                data = test_dataset[ind]
                data_time = time.time() - end
                points = data["points"]
                num_points = points.shape[0]

                # Convert points into tensor
                # torch.tensor always copy data
                points_batch = [t(torch.tensor(points, dtype=torch.float)) for t in transform_list]
                if cfg.TEST.VOTE.SHUFFLE:
                    index_batch = [torch.randperm(num_points) for _ in points_batch]
                    points_batch = [points[index] for points, index in zip(points_batch, index_batch)]
                points_batch = torch.stack(points_batch, dim=0).transpose_(1, 2).contiguous()
                points_batch = points_batch.cuda(non_blocking=True)

                preds = model({"points": points_batch})
                seg_logit_batch = preds["seg_logit"].cpu().numpy()

                if cfg.TEST.VOTE.SHUFFLE:
                    seg_logit_ensemble = np.zeros(seg_logit_batch.shape[1:], dtype=seg_logit_batch.dtype)
                    for i, index in enumerate(index_batch):
                        index = index.numpy()
                        seg_logit_ensemble[:, index] += seg_logit_batch[i]
                else:
                    seg_logit_ensemble = np.mean(seg_logit_batch, axis=0)

                seg_logit_all.append(seg_logit_ensemble)

                batch_time = time.time() - end
                end = time.time()

                if ind % cfg.TEST.LOG_PERIOD == 0:
                    logger.info("iter: {:4d}  time:{:.4f}  data:{:.4f}".format(ind, batch_time, data_time))
        # seg_logit_all = np.stack(seg_logit_all, axis=0)
    else:
        test_meters = MetricLogger(delimiter="  ")
        with torch.no_grad():
            start_time = time.time()
            end = start_time
            for iteration, data_batch in enumerate(test_data_loader):
                data_time = time.time() - end

                data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

                preds = model(data_batch)

                seg_logit_all.append(preds["seg_logit"].cpu().numpy())
                loss_dict = loss_fn(preds, data_batch)
                metric_dict = metric_fn(preds, data_batch)
                losses = sum(loss_dict.values())
                test_meters.update(loss=losses, **loss_dict, **metric_dict)

                batch_time = time.time() - end
                end = time.time()
                test_meters.update(time=batch_time, data=data_time)

                if iteration % cfg.TEST.LOG_PERIOD == 0:
                    logger.info(
                        test_meters.delimiter.join(
                            [
                                "iter: {iter:4d}",
                                "{meters}",
                            ]
                        ).format(
                            iter=iteration,
                            meters=str(test_meters),
                        )
                    )
        test_time = time.time() - start_time
        logger.info("Test {}  forward time: {:.2f}s".format(test_meters.summary_str, test_time))
        # seg_logit_all = np.concatenate(seg_logit_all, axis=0)
        seg_logit_all = [lg[0] for lg in seg_logit_all]

    # evaluate_semantic_segmentation(test_dataset, seg_logit_all, output_dir=output_dir, vis_dir=vis_dir)

    # consolidate semantic predictions and write to file

    # gt_all = np.stack([points['seg_label'] for points in test_dataset], axis=0)
    seg_pred_all = [np.argmax(lg, axis=0) for lg in seg_logit_all]
    np.savez(osp.join(output_dir, 'raw_preds'), *seg_pred_all)
    np.savez(osp.join(output_dir, 'raw_logits'), *seg_logit_all)
    # np.savetxt(osp.join(output_dir, 'raw_gt.txt'), gt_all, fmt='%d')

    class_map = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    eval_inds = list(range(1, len(class_map)+1))
    labels_dict = defaultdict(list)
    i = 0
    for scene_ind, scene in enumerate(test_dataset.scene_points_list):
        num_chunks = test_dataset.scene_sizes[scene_ind]
        votes_mat = np.full((num_chunks, scene.shape[0]), -1 * np.inf)
        preds_mat = np.zeros((num_chunks, scene.shape[0]), dtype=int)

        start_ind = i
        while i < len(test_dataset.whole_scene_index) and test_dataset.whole_scene_index[i] == scene_ind:
            points = test_dataset[i]
            indices = points['indices']
            votes_mat[i - start_ind, indices] = np.max(seg_logit_all[i][eval_inds], 0)
            preds_mat[i - start_ind, indices] = np.argmax(seg_logit_all[i][eval_inds], 0)
            i += 1

        top_votes = np.argmax(votes_mat, axis=0)
        preds = preds_mat[top_votes, np.arange(scene.shape[0])]

        fname = osp.join(output_dir, 'scannet_preds_{}.txt'.format(scene_ind))
        with open(fname, 'w') as f:
            for p in preds:
                f.write(str(class_map[p]))
                f.write('\n')
        

def main():
    args = parse_args()

    # Load the configuration
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

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

    assert cfg.TASK == "semantic_segmentation"
    test(cfg, output_dir)


if __name__ == "__main__":
    main()
