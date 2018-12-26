#!/usr/bin/env python
import argparse
import os.path as osp

import torch

from shaper_fewshot.config import cfg, purge_cfg
from shaper_fewshot.engine.tester import test
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
        "-s",
        "--suffix",
        dest="suffix",
        default="",
        help="suffix of output result files",
        type=str,
    )
    parser.add_argument(
        "--save-pred",
        action="store_true",
        help="whether to save predictions",
    )
    parser.add_argument(
        "--save-eval",
        action="store_true",
        help="whether to save evaluation results",
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

    # only support few-shot classification now
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

    logger = setup_logger("shaper", output_dir, prefix="test")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    test(cfg, output_dir, args.save_pred, args.save_eval, args.suffix)


if __name__ == "__main__":
    main()
