from __future__ import division

import logging
import os.path as osp
from collections import defaultdict

import numpy as np
import imageio
from tqdm import tqdm
from prettytable import PrettyTable

from shaper.utils.pc_util import point_cloud_three_views


def evaluate_classification(dataset, pred_labels,
                            output_dir="", vis_dir="", suffix=""):
    logger = logging.getLogger("shaper.evaluator.cls")
    logger.info("Start evaluating and visualize in {}".format(vis_dir))

    num_samples = len(dataset)
    class_names = dataset.classes
    assert len(pred_labels) == num_samples
    if suffix:
        suffix = "_" + suffix

    true_positive_per_class = defaultdict(int)
    num_positive_per_class = defaultdict(int)

    for ind in tqdm(range(num_samples)):
        data = dataset[ind]
        gt_label = int(data["cls_labels"])
        pred_label = int(pred_labels[ind])

        # Guarantee that seen classes are keys
        true_positive_per_class[gt_label] += (gt_label == pred_label)
        num_positive_per_class[gt_label] += 1

        if pred_label != gt_label and vis_dir:
            fname = "{:04d}_label_{}_pred_{}" + suffix
            fname = osp.join(vis_dir, fname).format(
                ind, class_names[gt_label], class_names[pred_label])

            points = data["points"]
            img = point_cloud_three_views(points)
            imageio.imwrite(fname + '.jpg', img)
            np.savetxt(fname + '.xyz', points, fmt="%.4f")

    # Overall accuracy
    true_positive = sum(true_positive_per_class.values())
    assert sum(num_positive_per_class.values()) == num_samples
    overall_acc = true_positive / num_samples
    logger.info("overall accuracy={:.2f}%".format(100.0 * overall_acc))

    # Average class accuracy
    acc_per_class = []
    table = PrettyTable(["Class", "Accuracy", "Correct", "Total"])
    for ind, class_name in enumerate(class_names):
        if ind in num_positive_per_class:  # seen class
            acc = true_positive_per_class[ind] / num_positive_per_class[ind]
            acc_per_class.append(acc)
            table.add_row([class_name, "{:.2f}".format(100.0 * acc),
                           true_positive_per_class[ind],
                           num_positive_per_class[ind]])
        else:
            table.add_row([class_name, 0, 0, 0])
    logger.info("average class accuracy={:.2f}%.\n{}".format(
        100.0 * np.mean(acc_per_class), table))


def evaluate_classification_with_keypoints(dataset, pred_labels, key_point_inds,
                                           output_dir="", vis_dir="", suffix=""):
    logger = logging.getLogger("shaper.evaluator.cls")
    logger.info("Start evaluating and visualize in {}".format(vis_dir))

    num_samples = len(dataset)
    class_names = dataset.classes
    assert len(pred_labels) == num_samples
    if suffix:
        suffix = "_" + suffix

    true_positive_per_class = defaultdict(int)
    num_positive_per_class = defaultdict(int)

    for ind in tqdm(range(num_samples)):
        data = dataset[ind]
        gt_label = int(data["cls_labels"])
        pred_label = int(pred_labels[ind])

        # Guarantee that seen classes are keys
        true_positive_per_class[gt_label] += (gt_label == pred_label)
        num_positive_per_class[gt_label] += 1

        if pred_label != gt_label and vis_dir:
            fname = "{:04d}_label_{}_pred_{}" + suffix
            fname = osp.join(vis_dir, fname).format(
                ind, class_names[gt_label], class_names[pred_label])

            points = data["points"]
            img = point_cloud_three_views(points)
            imageio.imwrite(fname + '.jpg', img)

            points_num = points.shape[0]
            point_color = np.ones((points_num, 3))

            curr_keypoint_inds = key_point_inds[ind, ...]
            point_color[curr_keypoint_inds, ...] = [1, 0, 0]

            points = np.concatenate((points, point_color), -1)
            np.savetxt(fname + '.xyz', points, fmt="%.4f")

    # Overall accuracy
    true_positive = sum(true_positive_per_class.values())
    assert sum(num_positive_per_class.values()) == num_samples
    overall_acc = true_positive / num_samples
    logger.info("overall accuracy={:.2f}%".format(100.0 * overall_acc))

    # Average class accuracy
    acc_per_class = []
    table = PrettyTable(["Class", "Accuracy", "Correct", "Total"])
    for ind, class_name in enumerate(class_names):
        if ind in num_positive_per_class:  # seen class
            acc = true_positive_per_class[ind] / num_positive_per_class[ind]
            acc_per_class.append(acc)
            table.add_row([class_name, "{:.2f}".format(100.0 * acc),
                           true_positive_per_class[ind],
                           num_positive_per_class[ind]])
        else:
            table.add_row([class_name, 0, 0, 0])
    logger.info("average class accuracy={:.2f}%.\n{}".format(
        100.0 * np.mean(acc_per_class), table))
