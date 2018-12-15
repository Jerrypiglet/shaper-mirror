import os
import os.path as osp
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze few-shot learning accuracy")
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
    parser.add_argument(
        "--dir",
        dest="dir",
        default="outputs/few_shot/pointnet_fewshot_target_cls_ftB",
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
        "--finish-line-number",
        dest="finish_line_number",
        default=250,
        help="Line number when finish training",
        type=int,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cross_num = args.cross_num

    dir = args.dir

    last_test_acc_list = []
    best_test_acc_list = []
    parsed_path_list = []
    for i in range(cross_num):
        path = dir + "_%d" % i

        if os.path.exists(path + '/val_acc.txt'):
            test_acc = np.loadtxt(path + '/val_acc.txt')
            if test_acc.shape[0] < args.finish_line_number:
                print('{} unfinished.'.format(path))
            else:
                best_test_acc_list.append(np.max(test_acc))
                last_test_acc_list.append(test_acc[-1])
                parsed_path_list.append(path)
        else:
            print('{} not parsed.'.format(path))

    print("parsed dirs: {}".format(len(parsed_path_list)))
    for i in range(len(parsed_path_list)):
        print(parsed_path_list[i])
    if len(parsed_path_list) > 1:
        print("last test accuracy: \n", last_test_acc_list)
        print("mean: {:.4f}, std: {:.4f}".format(
            np.mean(last_test_acc_list), np.std(last_test_acc_list, ddof=1)))
        print("best test accuracy: \n", best_test_acc_list)
        print("mean: {:.4f}, std: {:.4f}".format(
            np.mean(best_test_acc_list), np.std(best_test_acc_list, ddof=1)))


if __name__ == "__main__":
    main()
