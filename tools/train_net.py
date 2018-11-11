import argparse

import torch

from shaper.config import cfg
from shaper.engine.trainer import train
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
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
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
        if output_dir == "@":
            output_dir = args.config_file.replace("configs", "outputs")
        mkdir(output_dir)

    logger = setup_logger("shaper", output_dir, prefix="train")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    # logger.info("Collecting env info (might take some time)")
    # logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as fid:
        config_str = "\n" + fid.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg)

    # TODO: add test
    if not args.skip_test:
        pass


if __name__ == "__main__":
    main()
