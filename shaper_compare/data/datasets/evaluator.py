from __future__ import division

import logging
import os.path as osp
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable


def evaluate_classification(dataset, pred_labels, aux_preds=None,
                            output_dir="", vis_dir="", suffix=""):
    logger = logging.getLogger("shaper.evaluator.cls")
    logger.info("Start evaluating and visualize in {}".format(vis_dir))

    num_samples = dataset.get_total_target_num()
    class_names = dataset.classes
    assert len(pred_labels) == num_samples
    if suffix:
        suffix = "_" + suffix

    true_positive_per_class = defaultdict(int)
    num_positive_per_class = defaultdict(int)

    for ind in tqdm(range(num_samples)):
        # data = dataset[ind]
        gt_label = int(dataset.total_target_data_labels[ind])
        pred_label = int(pred_labels[ind])

        # Guarantee that seen classes are keys
        true_positive_per_class[gt_label] += (gt_label == pred_label)
        num_positive_per_class[gt_label] += 1

        if pred_label != gt_label and vis_dir:
            fname = "{:04d}_label_{}_pred_{}" + suffix
            fname = osp.join(vis_dir, fname).format(
                ind, class_names[gt_label], class_names[pred_label])

            # points = data["points"]
            points = dataset.total_target_data_points[ind]
            num_points = len(points)

            # three views
            # img = point_cloud_three_views(points)
            # imageio.imwrite(fname + '.jpg', img)

            # point clouds
            if aux_preds is not None and "key_point_inds" in aux_preds:
                point_colors = np.ones([num_points, 3], dtype=points.dtype)
                curr_keypoint_inds = aux_preds["key_point_inds"][ind, ...]
                point_colors[curr_keypoint_inds, ...] = [1, 0, 0]
                points = np.concatenate((points, point_colors), -1)

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

    return overall_acc, np.mean(acc_per_class)
