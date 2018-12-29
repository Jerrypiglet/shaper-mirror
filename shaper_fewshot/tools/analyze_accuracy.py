#!/usr/bin/env python
import argparse
import os.path as osp

import numpy as np
from prettytable import PrettyTable

from shaper.utils.io import read_pkl
from shaper.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze evaluation accuracy")
    parser.add_argument(
        "-d",
        "--dir",
        required=True,
        help="directory that contains eval files",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--file",
        default="eval.pkl",
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

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    prefix = args.prefix
    if not prefix:
        prefix = osp.splitext(args.file)[0]
    logger = setup_logger("shaper", args.dir, prefix=prefix)
    logger.info(args)

    # ---------------------------------------------------------------------------- #
    # Load evaluation results
    # ---------------------------------------------------------------------------- #
    overall_acc_collection = []
    # acc_per_class_collection = []
    num_tp_per_class_collection = []
    # num_gt_per_class_collection = []
    class_names = None
    for cross_index in range(args.num_cross):
        cross_dir = osp.join(args.dir, "cross_%d" % cross_index)
        overall_acc_list = []
        # acc_per_class_list = []
        num_tp_per_class_list = []
        # num_gt_per_class_list = []
        for replica_index in range(args.num_replicas):
            replica_dir = osp.join(cross_dir, "rep_%d" % replica_index)
            eval_file = osp.join(replica_dir, args.file)
            eval_results = read_pkl(eval_file)
            overall_acc_list.append(eval_results["overall_acc"])
            # acc_per_class_list.append(eval_results["acc_per_class"])
            num_tp_per_class_list.append(eval_results["num_tp_per_class"])
            # num_gt_per_class_list.append(eval_results["num_gt_per_class"])
            if class_names is None:
                class_names = eval_results["class_names"]
            else:
                assert all(x == y for x,y in zip(class_names, eval_results["class_names"]))
        overall_acc_collection.append(overall_acc_list)
        # acc_per_class_collection.append(acc_per_class_list)
        num_tp_per_class_collection.append(num_tp_per_class_list)
        # num_gt_per_class_collection.append(num_gt_per_class_list)

    # ---------------------------------------------------------------------------- #
    # Summarize class-wise divergence
    # ---------------------------------------------------------------------------- #
    titles = ["Class \ Support"] + [("%d" % ind) for ind in range(args.num_cross)] + ["Mean", "Std"]
    table1 = PrettyTable(titles)
    num_tp_std_list = []
    for class_ind, class_name in enumerate(class_names):
        # std of numbers of true positive
        num_tp_list = []
        for num_tp_per_class_list in num_tp_per_class_collection:
            num_tp_list.append([x[class_ind] for x in num_tp_per_class_list])
        num_tp_std = np.std(num_tp_list, axis=1, ddof=1)
        num_tp_std_list.append(num_tp_std)
        table1_row = ["{:s}".format(class_name)] + \
                     ["{:.1f}".format(x) for x in num_tp_std] + \
                     ["{:.1f}".format(np.mean(num_tp_std)),
                      "{:.1f}".format(np.std(num_tp_std, ddof=1))]
        table1.add_row(table1_row)
    num_tp_std_matrix = np.asarray(num_tp_std_list)
    # Mean
    std_mean = [np.mean(num_tp_std_matrix[:, cross_index]) for cross_index in range(args.num_cross)]
    table1_row = ["Mean"] + ["{:.2f}".format(x) for x in std_mean] + \
                 ["{:.2f}".format(np.mean(std_mean)), "{:.2f}".format(np.std(std_mean, ddof=1))]
    table1.add_row(table1_row)
    # Std
    std_std = [np.std(num_tp_std_matrix[:, cross_index], ddof=1) for cross_index in range(args.num_cross)]
    table1_row = ["Std"] + ["{:.2f}".format(x) for x in std_std] + \
                 ["{:.2f}".format(np.mean(std_std)), "{:.2f}".format(np.std(std_std, ddof=1))]
    table1.add_row(table1_row)

    logger.info("Class-wise divergence summary\n{}".format(table1))

    # ---------------------------------------------------------------------------- #
    # Summarize accuracy
    # ---------------------------------------------------------------------------- #
    titles = ["Replica \ Support"] + [("%d" % ind) for ind in range(args.num_cross)]
    table2 = PrettyTable(titles)

    for replica_index in range(args.num_replicas):
        table2_row = ["{:d}".format(replica_index)] + \
                    ["{:.2f}".format(overall_acc_collection[cross_index][replica_index] * 100.0)
                     for cross_index in range(args.num_cross)]
        table2.add_row(table2_row)
    # Mean
    table2_row = ["Mean"] + \
                ["{:.2f}".format(np.mean(overall_acc_collection[cross_index]) * 100.0)
                 for cross_index in range(args.num_cross)]
    table2.add_row(table2_row)
    # Std
    table2_row = ["Std"] + \
                ["{:.2f}".format(np.std(overall_acc_collection[cross_index], ddof=1) * 100.0)
                 for cross_index in range(args.num_cross)]
    table2.add_row(table2_row)

    logger.info("Accuracy summary\n{}".format(table2))


if __name__ == "__main__":
    main()
