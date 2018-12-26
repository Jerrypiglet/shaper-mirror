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
    # Summarize accuracy
    # ---------------------------------------------------------------------------- #
    titles = ["Replica\Cross"] + [("%d" % ind) for ind in range(args.num_cross)]
    table = PrettyTable(titles)

    overall_acc_collection = []
    for cross_index in range(args.num_cross):
        cross_dir = osp.join(args.dir, "cross_%d" % cross_index)
        overall_acc_list = []
        for replica_index in range(args.num_replicas):
            replica_dir = osp.join(cross_dir, "rep_%d" % replica_index)
            eval_file = osp.join(replica_dir, args.file)
            eval_results = read_pkl(eval_file)
            overall_acc_list.append(eval_results["overall_acc"])
        overall_acc_collection.append(overall_acc_list)
    #
    for replica_index in range(args.num_replicas):
        table_row = ["{:d}".format(replica_index)] + \
                    ["{:.2f}".format(overall_acc_collection[cross_index][replica_index] * 100.0)
                     for cross_index in range(args.num_cross)]
        table.add_row(table_row)
    # Mean
    table_row = ["Mean"] + \
                ["{:.2f}".format(np.mean(overall_acc_collection[cross_index]) * 100.0)
                 for cross_index in range(args.num_cross)]
    table.add_row(table_row)
    # Std
    table_row = ["Std"] + \
                ["{:.2f}".format(np.std(overall_acc_collection[cross_index], ddof=1) * 100.0)
                 for cross_index in range(args.num_cross)]
    table.add_row(table_row)

    logger.info("Accuracy summary\n{}".format(table))


if __name__ == "__main__":
    main()
