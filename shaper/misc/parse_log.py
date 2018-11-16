import os
import os.path as osp
import argparse
import glob

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
    args = parser.parse_args()
    return args


def parse_log_file(log_file, suffix=""):
    log_dir = osp.dirname(log_file)
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=suffix)

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


def main():
    args = parse_args()
    if args.log_file:
        parse_log_file(args.log_file)
    if args.log_dir:
        parse_log_dir(args.log_dir, update=args.update)


if __name__ == "__main__":
    main()
