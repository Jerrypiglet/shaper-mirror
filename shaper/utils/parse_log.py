import os.path as osp
import argparse

from tensorboardX import SummaryWriter


_KEYWORDS = ["loss", "acc"]


def parse_args():
    parser = argparse.ArgumentParser(description="Parse log to tensorboard")
    parser.add_argument(
        "-f",
        "--log-file",
        help="path to log file",
        type=str,
    )
    args = parser.parse_args()
    return args


def parse_log(log_file):
    log_dir = osp.dirname(log_file)
    writer = SummaryWriter(log_dir=log_dir)

    with open(log_file, 'r') as fid:
        lines = fid.readlines()
        lines = [line.strip() for line in lines if "shaper.trainer INFO: Epoch" in line]

    for line in lines:
        epoch = int(line[line.find('[')+1: line.find(']')])
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


def main():
    args = parse_args()
    parse_log(args.log_file)


if __name__ == "__main__":
    main()
