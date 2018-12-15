#!/usr/bin/env python
import argparse
import os.path as osp
import logging

import numpy as np
import scipy.stats as stats

import torch

from shaper_few_shot.config import cfg
from shaper_few_shot.engine.tester import test
from shaper.utils.logger import setup_logger, shutdown_logger
from shaper.utils.np_util import np_softmax
from shaper.utils.io import mkdir
from shaper.data.datasets import evaluate_classification


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch 3D Deep Learning Testing For Few-shot Ensemble")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="/home/rayc/Projects/shaper/configs/few_shot/pointnet_fewshot_target_cls_best.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--repeat_num",
        help="Repeat number for each support data",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--score_heur",
        choices=["logit", "softmax", "label"],
        default="logit",
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


def ensemble_test(args, cfg):
    num_gpus = torch.cuda.device_count()

    output_dir_ensemble = cfg.OUTPUT_DIR
    if output_dir_ensemble:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace("configs", "outputs")
        output_dir_ensemble = output_dir_ensemble.replace('@', config_path)
        mkdir(output_dir_ensemble)
    logger = setup_logger("shaper", output_dir_ensemble, prefix="ensemble_test")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    # logger.info("Collecting env info (might take some time)")
    # logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    # with open(args.config_file, "r") as fid:
    #     config_str = "\n" + fid.read()
    #     logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    dataset_collection = []
    cls_logits_collection = []
    overall_acc_collection = []
    acc_per_class_collection = []
    vis_dir = cfg.TEST.VIS_DIR.replace("@", output_dir_ensemble)
    if vis_dir:
        mkdir(vis_dir)

    for i in range(args.repeat_num):
        output_dir = cfg.OUTPUT_DIR + "/REP_%d" % i
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace("configs", "outputs")
        output_dir = output_dir.replace('@', config_path)
        assert osp.exists(output_dir), "Directory {} not exists.".format(output_dir)

        dataset, cls_logits, overall_acc, acc_per_class = test(cfg, output_dir)
        dataset_collection.append(dataset)
        cls_logits_collection.append(cls_logits)
        overall_acc_collection.append(overall_acc)
        acc_per_class_collection.append(acc_per_class)

    # multi model ensemble
    if cfg.TEST.VOTE.ENABLE:
        cls_logits_collection = np.concatenate(cls_logits_collection)
        if args.score_heur == "logit":
            cls_logits_ensemble = np.mean(cls_logits_collection, axis=0)
            pred_labels = np.argmax(cls_logits_ensemble, -1)  # (num_samples,)
        elif args.score_heur == "softmax":
            cls_probs_all = np_softmax(np.asarray(cls_logits_collection))
            cls_probs_ensemble = np.mean(cls_probs_all, axis=0)
            pred_labels = np.argmax(cls_probs_ensemble, -1)
        elif args.score_heur == "label":
            pred_labels_all = np.argmax(cls_logits_collection, -1)
            pred_labels = stats.mode(pred_labels_all, axis=0)[0].squeeze(0)
        else:
            raise ValueError("Unknown score heuristic")

        overall_acc, acc_per_class = evaluate_classification(
            dataset_collection[0], pred_labels,
            output_dir=output_dir_ensemble,
            vis_dir=vis_dir,
            suffix=args.score_heur,
        )

    else:
        cls_logits_collection = np.concatenate(cls_logits_collection)
        if args.score_heur == "logit":
            cls_logits_ensemble = np.mean(cls_logits_collection, axis=0)
            pred_labels = np.argmax(cls_logits_ensemble, -1)  # (num_samples,)
        elif args.score_heur == "softmax":
            cls_probs_all = np_softmax(np.asarray(cls_logits_collection))
            cls_probs_ensemble = np.mean(cls_probs_all, axis=0)
            pred_labels = np.argmax(cls_probs_ensemble, -1)
        elif args.score_heur == "label":
            pred_labels_all = np.argmax(cls_logits_collection, -1)
            pred_labels = stats.mode(pred_labels_all, axis=0)[0].squeeze(0)
        else:
            raise ValueError("Unknown score heuristic")

        overall_acc, acc_per_class = evaluate_classification(
            dataset_collection[0], pred_labels,
            output_dir=output_dir,
            vis_dir=vis_dir)

    logger.info("Overall accuracy for [{}] times repetition:".format(args.repeat_num))
    logger.info(",".join("{:.4f}".format(x) for x in overall_acc_collection))
    logger.info("min: {:.4f}, max: {:.4f}, mean: {:.4f}, std: {:.4f}".format(
        np.min(overall_acc_collection), np.max(overall_acc_collection),
        np.mean(overall_acc_collection), np.std(overall_acc_collection, ddof=1)))
    logger.info("Average class accuracy for [{}] times repetition:".format(args.repeat_num))
    logger.info(",".join("{:.4f}".format(x) for x in acc_per_class_collection))
    logger.info("min: {:.4f}, max: {:.4f}, mean: {:.4f}, std: {:.4f}".format(
        np.min(acc_per_class_collection), np.max(acc_per_class_collection),
        np.mean(acc_per_class_collection), np.std(acc_per_class_collection, ddof=1)))

    logger.info("Ensemble result: ")
    logger.info("Overall accuracy: {:.2f}%".format(100.0 * overall_acc))
    logger.info("Average class accuracy: {:.2f}%".format(100.0 * acc_per_class))

    shutdown_logger(logger)

    return overall_acc, acc_per_class, overall_acc_collection, acc_per_class_collection


def main():
    args = parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    ensemble_test(args, cfg)


if __name__ == "__main__":
    main()
