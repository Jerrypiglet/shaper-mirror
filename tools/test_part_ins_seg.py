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

from shaper.config.part_instance_segmentation import cfg
from shaper.config import purge_cfg
from shaper.models.build import build_model
from shaper.data.build import build_dataloader, build_transform
from shaper.data import transforms as T
from shaper.data.datasets.evaluator import evaluate_part_instance_segmentation
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
    conf_logit_all = []
    model.eval()
    loss_fn.eval()
    #metric_fn.eval()
    set_random_seed(cfg.RNG_SEED)

    if cfg.TEST.VOTE.NUM_VOTE > 1:
        # Disable inherent shuffle
        test_dataset.shuffle_points = False
        # Remove old transform
        test_dataset.transform = None

        if cfg.TEST.VOTE.TYPE == "AUGMENTATION":
            tmp_cfg = cfg.clone()
            tmp_cfg.defrost()
            tmp_cfg.TEST.AUGMENTATION = tmp_cfg.TEST.VOTE.AUGMENTATION
            transform_list = [build_transform(tmp_cfg, False)] * cfg.TEST.VOTE.NUM_VOTE
        elif cfg.TEST.VOTE.TYPE == "MULTI_VIEW":
            # Build new transform
            transform_list = []
            for view_ind in range(cfg.TEST.VOTE.NUM_VOTE):
                t = [T.PointCloudToTensor(),
                     T.PointCloudRotateByAngle(cfg.TEST.VOTE.MULTI_VIEW.AXIS,
                                               2 * np.pi * view_ind / cfg.TEST.VOTE.NUM_VOTE)]
                transform_list.append(T.Compose(t))
        else:
            raise NotImplementedError("Unsupported voting method.")

        with torch.no_grad():
            start_time = time.time()
            end = start_time
            for ind in range(len(test_dataset)):
                data = test_dataset[ind]
                data_time = time.time() - end
                points = data["points"]
                cls_label = data["cls_label"]
                num_points = points.shape[0]

                # Convert points into tensor
                points_batch = [t(points) for t in transform_list]
                if cfg.TEST.VOTE.SHUFFLE:
                    index_batch = [torch.randperm(num_points) for _ in points_batch]
                    points_batch = [points[index] for points, index in zip(points_batch, index_batch)]
                points_batch = torch.stack(points_batch, dim=0).transpose_(1, 2).contiguous()
                points_batch = points_batch.cuda(non_blocking=True)
                cls_label_batch = torch.tensor([cls_label] * cfg.TEST.VOTE.NUM_VOTE).cuda()

                preds = model({"points": points_batch, "cls_label": cls_label_batch})
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
        seg_logit_all = np.stack(seg_logit_all, axis=0)
    else:
        test_meters = MetricLogger(delimiter="  ")
        with torch.no_grad():
            start_time = time.time()
            end = start_time
            for iteration, data_batch in enumerate(test_data_loader):
                data_time = time.time() - end

                data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

                preds = model(data_batch)

                seg_logit_all.append(preds["mask_output"].cpu().numpy())
                conf_logit_all.append(preds["global_output"].cpu().numpy())
                loss_dict = loss_fn(preds, data_batch)
                #metric_dict = metric_fn(preds, data_batch)
                losses = sum(loss_dict.values())
                test_meters.update(loss=losses, **loss_dict)

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
        seg_logit_all = np.concatenate(seg_logit_all, axis=0)
        conf_logit_all = np.concatenate(conf_logit_all, axis=0)

    evaluate_part_instance_segmentation(test_dataset, seg_logit_all, conf_logit_all,output_dir=output_dir, vis_dir=vis_dir)


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

    assert cfg.TASK == "part_instance_segmentation"
    test(cfg, output_dir)


if __name__ == "__main__":
    main()
