import os
import os.path as osp
import argparse
import glob

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

# _KEYWORDS = ["loss", "acc"]
_KEYWORDS = ["acc"]


def parse_args():
    parser = argparse.ArgumentParser(description="Parse log to tensorboard")
    parser.add_argument(
        "-f",
        "--log-file",
        help="path to log file",
        default="/home/rayc/Projects/shaper/outputs/few_shot/pn2ssg/"
                "pointnet2ssg_fewshot_target_cls_ap_baseline_scratch/REP_0/log.train.12_02_09_59_24.txt",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--log-dir",
        help="path to log directory",
        type=str,
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="whether to update tf events",
    )
    parser.add_argument(
        "--dir",
        dest="dir",
        default="/home/rayc/Projects/shaper/outputs/few_shot/pointnet2ssg_fewshot_target_cls_last/last_500EP",
        help="Output directory",
        type=str,
    )

    parser.add_argument(
        "--cross-num",
        dest="cross_num",
        default=10,
        help="Total cross number",
        type=int,
    )

    parser.add_argument(
        "--prefix",
        default="",
        help="Prefix for chosen log file",
        type=str,
    )

    parser.add_argument(
        "--save-img",
        dest="save_img",
        default=True,
        help="Whether to save accuracy curve",
        type=bool
    )

    args = parser.parse_args()
    return args


def parse_log_file(log_file, suffix=""):
    log_dir = osp.dirname(log_file)
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=suffix)

    with open(log_file, 'r') as fid:
        lines = fid.readlines()
        lines = [line.strip() for line in lines if "shaper.trainer INFO: Epoch" in line]

    for line in lines:
        epoch = int(line[line.find('[') + 1: line.find(']')])
        if "-Train" in line:
            metric_str = line.split("-Train ")[-1]
            prefix = "train"
        elif "-Val" in line:
            metric_str = line.split("-Val ")[-1]
            prefix = "val"
        else:
            raise ValueError()

        for meter in metric_str.split("  "):
            try:
                k, v = meter.split(":")
            except:
                continue
            for keyword in _KEYWORDS:
                if keyword in k:
                    writer.add_scalar(osp.join(prefix, k), float(v), global_step=epoch)

    writer.close()


def parse_log_dir(log_dir, pattern="log.train*.txt", update=False):
    for sub_dir in glob.glob(osp.join(log_dir, "**")):
        log_files = glob.glob(osp.join(sub_dir, pattern))
        if log_files:
            event_files = glob.glob(osp.join(sub_dir, "events.out.tfevents*"))
            if update:
                for event_file in event_files:
                    print("Deleting {}".format(event_file))
                    os.remove(event_file)
            elif event_files:
                print("Exists old events at {}".format(event_files))
                return
            for ind, log_file in enumerate(log_files):
                parse_log_file(log_file, suffix=str(ind))


def parse_log_file_to_txt(log_file, suffix="", save_img=True):
    log_dir = osp.dirname(log_file)
    print("log_dir: ", log_dir)
    if suffix == "":
        train_file_writer = open(osp.join(log_dir, "train_acc.txt"), "w")
        val_file_writer = open(osp.join(log_dir, "val_acc.txt"), "w")
    else:
        train_file_writer = open(osp.join(log_dir, "train_acc_" + suffix + ".txt"), "w")
        val_file_writer = open(osp.join(log_dir, "val_acc_" + suffix + ".txt"), "w")

    train_acc = []
    val_acc = []

    with open(log_file, 'r') as fid:
        lines = fid.readlines()
        lines = [line.strip() for line in lines if "shaper.trainer INFO: Epoch" in line]

    for line in lines:
        epoch = int(line[line.find('[') + 1: line.find(']')])
        if "-Train" in line:
            metric_str = line.split("-Train ")[-1]
            prefix = "train"
        elif "-Val" in line:
            metric_str = line.split("-Val ")[-1]
            prefix = "val"
        else:
            raise ValueError()

        for meter in metric_str.split("  "):
            try:
                k, v = meter.split(":")
            except:
                continue
            for keyword in _KEYWORDS:
                if keyword in k:
                    if prefix == "train":
                        train_file_writer.write("{:.4f}\n".format(float(v)))
                        train_acc.append(float(v))
                    else:
                        val_file_writer.write("{:.4f}\n".format(float(v)))
                        val_acc.append(float(v))

    train_file_writer.close()
    val_file_writer.close()

    if save_img:
        train_acc = np.array(train_acc)
        val_acc = np.array(val_acc)
        plt.figure()
        plt.plot(train_acc, color='blue', label='Train Acc')
        plt.plot(val_acc, color='red', linewidth=1.0, linestyle='--', label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.savefig(osp.join(log_dir, 'acc.png'))


def batch_parse_log_txt(dir, cross_num, prefix, output_suffix=""):
    for i in range(cross_num):
        path = dir + "_%d" % i
        file = glob.glob(osp.join(path, prefix + "*log.train.*.txt"))
        if file:
            print(file[0])
            parse_log_file_to_txt(file[0], output_suffix)  # choose the first


def main():
    args = parse_args()
    if args.log_file:
        parse_log_file_to_txt(args.log_file)
    # if args.log_dir:
    #     parse_log_dir(args.log_dir, update=args.update)


if __name__ == "__main__":
    # main()
    args = parse_args()
    if args.log_file:
        parse_log_file_to_txt(args.log_file)
    else:
        batch_parse_log_txt(args.dir, args.cross_num, args.prefix)
