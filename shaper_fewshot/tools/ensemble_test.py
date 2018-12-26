#!/usr/bin/env python
import argparse
import os.path as osp

import numpy as np
import scipy.stats as stats

from shaper_fewshot.config import cfg, purge_cfg
from shaper_fewshot.data.build import build_dataset
from shaper.data.datasets.evaluator import evaluate_classification
from shaper.utils.io import read_pkl
from shaper.utils.logger import setup_logger
from shaper.utils.np_util import np_softmax


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble test")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--dir",
        required=True,
        help="directory that contains predictions",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--file",
        default="pred.pkl",
        help="name of eval file",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--prefix",
        help="prefix of log file",
        type=str,
    )
    parser.add_argument(
        "--num-cross",
        default=10,
        help="number of cross",
        type=int,
    )
    parser.add_argument(
        "--num-replicas",
        default=10,
        help="number of replicas",
        type=int,
    )
    parser.add_argument(
        "--score-heur",
        default="logit",
        help="heuristic to ensemble score",
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


def ensemble_test(cls_logit_collection, score_heur="logit"):
    if score_heur == "logit":
        cls_logit_ensemble = np.mean(cls_logit_collection, axis=0)
        pred_labels = np.argmax(cls_logit_ensemble, -1)  # (num_samples,)
    elif score_heur == "softmax":
        cls_prob_collection = np_softmax(np.asarray(cls_logit_collection))
        cls_prob_ensemble = np.mean(cls_prob_collection, axis=0)
        pred_labels = np.argmax(cls_prob_ensemble, -1)
    elif score_heur == "label":
        pred_label_collection = np.argmax(cls_logit_collection, -1)
        pred_labels = stats.mode(pred_label_collection, axis=0)[0].squeeze(0)
    else:
        raise ValueError("Unknown score heuristic")
    return pred_labels


def main():
    args = parse_args()

    # only support few-shot classification now
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    prefix = args.prefix
    if not prefix:
        prefix = "ensemble_" + osp.splitext(args.file)[0]
    logger = setup_logger("shaper", args.dir, prefix=prefix)
    logger.info(args)

    # ---------------------------------------------------------------------------- #
    # Collect to ensemble
    # ---------------------------------------------------------------------------- #
    # Only support ModelNet now
    dataset = build_dataset(cfg, mode="test")
    eval_results_collection = []
    for cross_index in range(args.num_cross):
        cross_dir = osp.join(args.dir, "cross_%d" % cross_index)
        cls_logit_collection = []
        for replica_index in range(args.num_replicas):
            replica_dir = osp.join(cross_dir, "rep_%d" % replica_index)
            pred_file = osp.join(replica_dir, args.file)
            pred_list = read_pkl(pred_file)  # list of dict
            cls_logit_collection.extend([pred["cls_logit"] for pred in pred_list])
        pred_labels = ensemble_test(cls_logit_collection, args.score_heur)
        eval_results = evaluate_classification(dataset, pred_labels)
        eval_results_collection.append(eval_results)

    overall_acc_collection = [eval_results["overall_acc"] for eval_results in eval_results_collection]
    logger.info("Overall accuracy: mean={:.2f}, std={:.2f}".format(
        np.mean(overall_acc_collection) * 100.0, np.std(overall_acc_collection, ddof=1) * 100.0))
    logger.info(','.join("{:.2f}".format(overall_acc * 100.0) for overall_acc in overall_acc_collection))


if __name__ == "__main__":
    main()
