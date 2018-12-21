#!/usr/bin/env python
# test_net.py is almost same to train_net.py;
# however, it is written for possible discrepancy of functions.
import argparse
import os.path as osp
import importlib

import torch

from shaper.config import purge_cfg
from shaper.engine.tester import test
from shaper.utils.io import mkdir
from shaper.utils.logger import setup_logger


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


def main():
    args = parse_args()
    num_gpus = torch.cuda.device_count()

    cfg = importlib.import_module("shaper.config.{:s}".format(args.task)).cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg = purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace("configs", "outputs")
        output_dir = output_dir.replace('@', config_path)
        mkdir(output_dir)

    logger = setup_logger("shaper", output_dir, prefix="test")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    test(cfg, output_dir)


if __name__ == "__main__":
    main()
