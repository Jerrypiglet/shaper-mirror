import argparse
import os.path as osp
import numpy as np

from prettytable import PrettyTable

from shaper.utils.io import mkdir
from shaper.utils.logger import setup_logger, shutdown_logger
from shaper_compare.tools.ensemble_compare_test_net import cfg, ensemble_test


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch 3D Deep Learning Testing For CompareNet Ensemble")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="/home/rayc/Projects/shaper/configs/modelnet/comparenet/pointnet/pointnet_target_cls_best.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--cross_num",
        help="Number of support datasets",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--repeat_num",
        help="Repeat number for each support data",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--score_heur",
        choices=["soft_label", "label"],
        default="soft_label",
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


if __name__ == "__main__":
    args = parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    OUTPUT_DIR = "@"
    output_dir_batch_ensemble = OUTPUT_DIR
    if output_dir_batch_ensemble:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace("configs", "outputs")
        output_dir_batch_ensemble = output_dir_batch_ensemble.replace('@', config_path)
        mkdir(output_dir_batch_ensemble)
    logger = setup_logger("batch_ensemble_test", output_dir_batch_ensemble, prefix="batch_ensemble_test")
    logger.info("Running with args: \n{}".format(args))

    logger.info("Loaded configuration file {}".format(args.config_file))

    logger.info("Running with config:\n{}".format(cfg))

    overall_acc_list = []
    per_class_acc_list = []
    overall_acc_collection_list = []
    per_class_acc_collection_list = []

    cross_num = args.cross_num
    if cross_num > 0:
        for i in range(cross_num):
            cfg.OUTPUT_DIR = OUTPUT_DIR + "/cross_%d" % i
            cfg.DATASET.COMPARE.CROSS_NUM = i
            overall_acc, per_class_acc, overall_acc_collection, per_class_acc_collection = ensemble_test(args, cfg)
            overall_acc_list.append(overall_acc)
            per_class_acc_list.append(per_class_acc)
            overall_acc_collection_list.append(overall_acc_collection)
            per_class_acc_collection_list.append(per_class_acc_collection)
        logger.info(output_dir_batch_ensemble)
        logger.info("last test accuracy of [{}] support datasets: ".format(cross_num))
        logger.info(",".join("{:.4f}".format(x) for x in overall_acc_list))
        logger.info("mean: {:.4f}, std: {:.4f}".format(
            np.mean(overall_acc_list), np.std(overall_acc_list, ddof=1)
        ))
        table_items = ["Support"]
        for i in range(cross_num):
            table_items.append("%d" % i)
        table = PrettyTable(table_items)
        ensemble_row = ["ensemble"]
        for i in range(cross_num):
            ensemble_row.append("{:.4f}".format(overall_acc_list[i]))
        table.add_row(ensemble_row)
        for j in range(args.repeat_num):
            row = ["%d" % j]
            for i in range(cross_num):
                row.append("{:.4f}".format(overall_acc_collection_list[i][j]))
            table.add_row(row)
        mean_row = ["mean"]
        for i in range(cross_num):
            mean_row.append("{:.4f}".format(np.mean(overall_acc_collection_list[i])))
        table.add_row(mean_row)
        std_row = ["std"]
        for i in range(cross_num):
            std_row.append("{:.4f}".format(np.std(overall_acc_collection_list[i], ddof=1)))
        table.add_row(std_row)
        logger.info("\n{}".format(table))
    else:
        cross_num = 1
        overall_acc, per_class_acc, overall_acc_collection, per_class_acc_collection = ensemble_test(args, cfg)
        overall_acc_list.append(overall_acc)
        per_class_acc_list.append(per_class_acc)
        overall_acc_collection_list.append(overall_acc_collection)
        logger.info(output_dir_batch_ensemble)
        logger.info("last test accuracy of [{}] support datasets: ".format(cross_num))
        logger.info(",".join("{:.4f}".format(x) for x in overall_acc_list))
        logger.info("mean: {:.4f}".format(
            np.mean(overall_acc_list)
        ))
        table_items = ["Support"]
        for i in range(cross_num):
            table_items.append("%d" % i)
        table = PrettyTable(table_items)
        ensemble_row = ["ensemble"]
        for i in range(cross_num):
            ensemble_row.append("{:.4f}".format(overall_acc_list[i]))
        table.add_row(ensemble_row)
        for j in range(args.repeat_num):
            row = ["%d" % j]
            for i in range(cross_num):
                row.append("{:.4f}".format(overall_acc_collection_list[i][j]))
            table.add_row(row)
        mean_row = ["mean"]
        for i in range(cross_num):
            mean_row.append("{:.4f}".format(np.mean(overall_acc_collection_list[i])))
        table.add_row(mean_row)
        std_row = ["std"]
        for i in range(cross_num):
            std_row.append("{:.4f}".format(np.std(overall_acc_collection_list[i], ddof=1)))
        table.add_row(std_row)
        logger.info("\n{}".format(table))

    shutdown_logger(logger)
