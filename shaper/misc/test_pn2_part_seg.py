#!/usr/bin/env python
# test_cls.py is almost same to train_cls.py;
# however, it is written for possible discrepancy of functions.
import argparse
import os.path as osp
import importlib
import logging
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from shaper.config import purge_cfg
from shaper.models import build_model
from shaper.data import build_dataloader
from shaper.utils.checkpoint import Checkpointer
from shaper.utils.logger import setup_logger
from shaper.utils.io import mkdir


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
        "-t",
        "--task",
        dest="task",
        default="classification",
        help="task to train or test",
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

    # build model
    model, loss_fn, metric_fn = build_model(cfg)
    model = nn.DataParallel(model).cuda()

    # build checkpointer
    checkpointer = Checkpointer(model, save_dir=output_dir)

    if cfg.TEST.WEIGHT:
        weight_path = cfg.TEST.WEIGHT.replace("@", output_dir)
        checkpointer.load(weight_path, resume=False)
    else:
        checkpointer.load(None, resume=True)

    # build data loader
    test_data_loader = build_dataloader(cfg, mode="test")
    test_dataset = test_data_loader.dataset
    class_to_seg_map = test_dataset.class_to_seg_map

    # test
    model.eval()
    num_inst_per_class = defaultdict(int)
    iou_per_class = defaultdict(float)

    with torch.no_grad():
        for iteration, data_batch in enumerate(test_data_loader):
            cls_labels = data_batch['cls_label'].numpy()
            seg_labels = data_batch['seg_label'].numpy()
            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

            preds = model(data_batch)

            pred_seg_logits = preds['seg_logit'].cpu().numpy()

            # IOU
            for cls_label, seg_label, pred_seg_logit in zip(cls_labels, seg_labels, pred_seg_logits):
                assert pred_seg_logit.shape[1] == seg_label.shape[0]
                segids = class_to_seg_map[cls_label]
                pred_seg_logit = pred_seg_logit[segids, :]
                gt_seg_label = seg_label

                pred_seg_label = np.argmax(pred_seg_logit, axis=0)
                for ind, segid in enumerate(segids):
                    # convert partid to segid
                    pred_seg_label[pred_seg_label == ind] = segid

                tp_mask = (pred_seg_label == gt_seg_label)
                num_inst_per_class[cls_label] += 1

                iou_per_instance = 0.0
                for ind, segid in enumerate(segids):
                    gt_mask = (gt_seg_label == segid)
                    num_intersection = np.sum(np.logical_and(tp_mask, gt_mask))
                    num_pos = np.sum(pred_seg_label == segid)
                    num_gt = np.sum(gt_mask)
                    num_union = num_pos + num_gt - num_intersection
                    iou = num_intersection / num_union if num_union > 0 else 1.0
                    iou_per_instance += iou
                iou_per_instance /= len(segids)
                iou_per_class[cls_label] += iou_per_instance

        total_iou = sum(iou_per_class.values())
        overall_iou = total_iou / len(test_dataset)
        logger.info("overall IOU={:.2f}".format(100.0 * overall_iou))


def main():
    args = parse_args()
    num_gpus = torch.cuda.device_count()

    cfg = importlib.import_module("shaper.config.{:s}".format(args.task)).cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace("configs", "outputs")
        output_dir = output_dir.replace('@', config_path)
        mkdir(output_dir)

    logger = setup_logger("shaper", output_dir, prefix="test_part_seg")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    test(cfg, output_dir)


if __name__ == "__main__":
    main()
