from __future__ import division

import logging
import os.path as osp
from collections import defaultdict

import numpy as np
import imageio
from tqdm import tqdm
from prettytable import PrettyTable

from shaper.utils.pc_util import point_cloud_three_views
from shaper.data.datasets.visualize import gen_visu
import os


def evaluate_classification(dataset,
                            pred_labels,
                            aux_preds=None,
                            output_dir="",
                            vis_dir="",
                            suffix=""):
    """Evaluate classification results

    Args:
        dataset (torch.utils.data.Dataset): dataset
        pred_labels (list of int or np.ndarray): predicted labels
        aux_preds (dict, optional): auxiliary predictions
        output_dir (str, optional): output directory
        vis_dir (str, optional): visualization directory
        suffix (str, optional):

    """
    logger = logging.getLogger("shaper.evaluator.cls")
    logger.info("Start evaluating and visualize in {}".format(vis_dir))

    # Remove transform
    dataset.transform = None
    # Use all points
    dataset.num_points = -1
    dataset.shuffle_points = False

    # aliases
    num_samples = len(dataset)
    class_names = dataset.classes
    assert len(pred_labels) == num_samples
    if suffix:
        suffix = "_" + suffix

    num_tp_per_class = defaultdict(int)
    # The number of ground_truth
    num_gt_per_class = defaultdict(int)

    for ind in tqdm(range(num_samples)):
        data = dataset[ind]
        gt_label = int(data["cls_label"])
        pred_label = int(pred_labels[ind])

        # Guarantee that seen classes are keys
        num_tp_per_class[gt_label] += (gt_label == pred_label)
        num_gt_per_class[gt_label] += 1

        if pred_label != gt_label and vis_dir:
            fname = "{:04d}_label_{}_pred_{}" + suffix
            fname = osp.join(vis_dir, fname).format(ind, class_names[gt_label], class_names[pred_label])

            points = data["points"]
            num_points = len(points)

            # three views
            img = point_cloud_three_views(points)
            imageio.imwrite(fname + '.jpg', img)

            # point clouds
            if aux_preds is not None and "key_point_inds" in aux_preds:
                point_colors = np.ones([num_points, 3], dtype=points.dtype)
                # TODO: remove invalid index
                key_point_inds = aux_preds["key_point_inds"][ind]
                point_colors[key_point_inds, ...] = [1, 0, 0]
                points = np.concatenate((points, point_colors), -1)

            np.savetxt(fname + '.xyz', points, fmt="%.4f")

    # Overall accuracy
    total_tp = sum(num_tp_per_class.values())
    assert sum(num_gt_per_class.values()) == num_samples
    overall_acc = total_tp / num_samples
    logger.info("overall accuracy={:.2f}%".format(100.0 * overall_acc))

    # Average class accuracy
    acc_per_class = []
    table = PrettyTable(["Class", "Accuracy", "Correct", "Total"])
    for ind, class_name in enumerate(class_names):
        if ind in num_gt_per_class:  # seen class
            acc = num_tp_per_class[ind] / num_gt_per_class[ind]
            acc_per_class.append(acc)
            table.add_row([class_name, "{:.2f}".format(100.0 * acc),
                           num_tp_per_class[ind],
                           num_gt_per_class[ind]])
        else:
            table.add_row([class_name, 0, 0, 0])
    logger.info("average class accuracy={:.2f}%.\n{}".format(
        100.0 * np.mean(acc_per_class), table))

    return {"overall_acc": overall_acc,
            "acc_per_class": acc_per_class,
            "num_tp_per_class": num_tp_per_class,
            "num_gt_per_class": num_gt_per_class,
            "class_names": class_names,
            }


def evaluate_part_segmentation(dataset,
                               pred_logits,
                               aux_preds=None,
                               output_dir="",
                               vis_dir="",
                               suffix=""):
    """Evaluate part segmentation results

    Args:
        dataset (torch.utils.data.Dataset): dataset
        pred_logits (list of np.ndarray or np.ndarray): predicted logits
        aux_preds (dict, optional): auxiliary predictions
        output_dir (str, optional): output directory
        vis_dir (str, optional): visualization directory
        suffix (str, optional):

    """
    logger = logging.getLogger("shaper.evaluator.part_seg")
    logger.info("Start evaluating and visualize in {}".format(vis_dir))

    # Remove transform
    dataset.transform = None
    # Use all points
    dataset.num_points = -1
    dataset.shuffle_points = False

    # aliases
    num_samples = len(dataset)
    class_names = dataset.classes
    class_to_seg_map = dataset.class_to_seg_map
    assert len(pred_logits) == num_samples

    seg_acc_per_class = defaultdict(float)
    num_inst_per_class = defaultdict(int)
    iou_per_class = defaultdict(float)

    for ind in tqdm(range(num_samples)):
        data = dataset[ind]
        points = data["points"]
        gt_cls_label = data["cls_label"]
        gt_seg_label = data["seg_label"]
        # (num_seg_classes, num_points)
        pred_seg_logit = pred_logits[ind]

        # sanity check
        assert len(gt_seg_label) == points.shape[0]
        # assert pred_seg_logit.shape[1] >= points.shape[0]

        segids = class_to_seg_map[gt_cls_label]
        num_valid_points = min(pred_seg_logit.shape[1], points.shape[0])
        pred_seg_logit = pred_seg_logit[segids, :num_valid_points]
        # pred_seg_logit = pred_seg_logit[:, :num_valid_points]
        gt_seg_label = gt_seg_label[:num_valid_points]

        pred_seg_label = np.argmax(pred_seg_logit, axis=0)
        for ind, segid in enumerate(segids):
            # convert partid to segid
            pred_seg_label[pred_seg_label == ind] = segid

        tp_mask = (pred_seg_label == gt_seg_label)
        seg_acc = np.mean(tp_mask)
        seg_acc_per_class[gt_cls_label] += seg_acc
        num_inst_per_class[gt_cls_label] += 1

        iou_per_instance = 0.0
        for ind, segid in enumerate(segids):
            gt_mask = (gt_seg_label == segid)
            num_intersection = np.sum(np.logical_and(tp_mask, gt_mask))
            num_pos = np.sum(pred_seg_label == segid)
            num_gt = np.sum(gt_mask)
            num_union = num_pos + num_gt - num_intersection
            iou = num_intersection / num_union if num_union > 0 else 1.0
            iou_per_instance += iou
        iou_per_instance /= len(segids)
        iou_per_class[gt_cls_label] += iou_per_instance

    # Overall
    total_seg_acc = sum(seg_acc_per_class.values())
    assert sum(num_inst_per_class.values()) == num_samples
    overall_acc = total_seg_acc / num_samples
    logger.info("overall segmentation accuracy={:.2f}%".format(100.0 * overall_acc))
    total_iou = sum(iou_per_class.values())
    overall_iou = total_iou / num_samples
    logger.info("overall IOU={:.2f}".format(100.0 * overall_iou))

    # Per class
    table = PrettyTable(["Class", "SegAccuracy", "IOU", "Total"])
    for ind, class_name in enumerate(class_names):
        if ind in num_inst_per_class:  # seen class
            seg_acc = seg_acc_per_class[ind] / num_inst_per_class[ind]
            iou = iou_per_class[ind] / num_inst_per_class[ind]
            table.add_row([class_name,
                           "{:.2f}".format(100.0 * seg_acc),
                           "{:.2f}".format(100.0 * iou),
                           num_inst_per_class[ind]])
        else:
            table.add_row([class_name, 0, 0, 0])
    logger.info("class-wise segmentation accuracy.\n{}".format(table))




def evaluate_part_instance_segmentation(dataset,
                               pred_logits,
                               conf_logits,
                               aux_preds=None,
                               output_dir="",
                               vis_dir="",
                               suffix=""):
    """Evaluate part segmentation results

    Args:
        dataset (torch.utils.data.Dataset): dataset
        pred_logits (list of np.ndarray or np.ndarray): predicted logits
        aux_preds (dict, optional): auxiliary predictions
        output_dir (str, optional): output directory
        vis_dir (str, optional): visualization directory
        suffix (str, optional):

    """
    logger = logging.getLogger("shaper.evaluator.part_seg")
    logger.info("Start evaluating and visualize in {}".format(vis_dir))

    # Remove transform
    dataset.transform = None
    # Use all points
    #dataset.num_points = -1
    dataset.shuffle_points = False

    # aliases
    num_samples = len(dataset)
    assert len(pred_logits) == num_samples

    seg_acc_per_class = defaultdict(float)
    num_inst_per_class = defaultdict(int)
    iou_per_class = defaultdict(float)

    for ind in tqdm(range(num_samples)):
        continue
        data = dataset[ind]
        points = data["points"]
        gt_seg_label = data["ins_seg_label"]
        # (num_seg_classes, num_points)
        pred_seg_logit = pred_logits[ind]

        # sanity check
        assert gt_seg_label.shape[1] == points.shape[0]
        # assert pred_seg_logit.shape[1] >= points.shape[0]

        num_valid_points = min(pred_seg_logit.shape[1], points.shape[0])
        pred_seg_logit = pred_seg_logit[segids, :num_valid_points]
        # pred_seg_logit = pred_seg_logit[:, :num_valid_points]
        gt_seg_label = gt_seg_label[:num_valid_points]

        pred_seg_label = np.argmax(pred_seg_logit, axis=0)
        for ind, segid in enumerate(segids):
            # convert partid to segid
            pred_seg_label[pred_seg_label == ind] = segid

        iou_per_instance = 0.0
        for ind, segid in enumerate(segids):
            gt_mask = (gt_seg_label == segid)
            num_intersection = np.sum(np.logical_and(tp_mask, gt_mask))
            num_pos = np.sum(pred_seg_label == segid)
            num_gt = np.sum(gt_mask)
            num_union = num_pos + num_gt - num_intersection
            iou = num_intersection / num_union if num_union > 0 else 1.0
            iou_per_instance += iou
        iou_per_instance /= len(segids)
        iou_per_class[gt_cls_label] += iou_per_instance


    gen_visu(os.path.join(output_dir,vis_dir), dataset, pred_logits, conf_logits)

    # Overall
    #total_seg_acc = sum(seg_acc_per_class.values())
    #assert sum(num_inst_per_class.values()) == num_samples
    #overall_acc = total_seg_acc / num_samples
    #logger.info("overall segmentation accuracy={:.2f}%".format(100.0 * overall_acc))
    #total_iou = sum(iou_per_class.values())
    #overall_iou = total_iou / num_samples
    #logger.info("overall IOU={:.2f}".format(100.0 * overall_iou))

    ## Per class
    #table = PrettyTable(["Class", "SegAccuracy", "IOU", "Total"])
    #for ind, class_name in enumerate(class_names):
    #    if ind in num_inst_per_class:  # seen class
    #        seg_acc = seg_acc_per_class[ind] / num_inst_per_class[ind]
    #        iou = iou_per_class[ind] / num_inst_per_class[ind]
    #        table.add_row([class_name,
    #                       "{:.2f}".format(100.0 * seg_acc),
    #                       "{:.2f}".format(100.0 * iou),
    #                       num_inst_per_class[ind]])
    #    else:
    #        table.add_row([class_name, 0, 0, 0])
    #logger.info("class-wise segmentation accuracy.\n{}".format(table))
