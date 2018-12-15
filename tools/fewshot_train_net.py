import argparse
import os.path as osp

import torch

from shaper_few_shot.config import cfg
from shaper_few_shot.engine.trainer import train
from shaper_few_shot.engine.tester import test
from shaper.utils.io import mkdir
from shaper.utils.logger import setup_logger


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
        "--do-test",
        dest="do_test",
        help="Test the final model",
        action="store_true",
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

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace("configs", "outputs")
        output_dir = output_dir.replace('@', config_path)
        mkdir(output_dir)

    logger = setup_logger("shaper", output_dir, prefix="train")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    # logger.info("Collecting env info (might take some time)")
    # logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    # with open(args.config_file, "r") as fid:
    #     config_str = "\n" + fid.read()
    #     logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, output_dir)

    if args.do_test:
        torch.cuda.empty_cache()
        test(cfg, output_dir)


if __name__ == "__main__":
    main()
