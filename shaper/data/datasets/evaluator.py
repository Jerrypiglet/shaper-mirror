from __future__ import division

import logging
import os.path as osp
from collections import defaultdict

import numpy as np
import imageio
from tqdm import tqdm
from prettytable import PrettyTable

from shaper.utils.pc_util import point_cloud_three_views
from shaper.data.datasets.visualize import gen_visu, gen_foveal_visu
import os
import torch


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

    #gt_masks= dataset.cache_ins_seg_label[:,:,:dataset.num_points]
    #point2group = np.concatenate([np.zeros((num_samples,2500)),dataset.cache_point2group], 1)
    #point2group = torch.tensor(point2group[:,:dataset.num_points]).long()
    #vm = torch.zeros((num_samples, 6,2500)).float()
    #pm = torch.zeros((num_samples, 6,2500)).float()
    #newp2g=point2group.unsqueeze(1).expand(pred_logits.shape)
    #pm = pm.scatter_add(2, newp2g, torch.tensor(pred_logits)).numpy()
    #vm = vm.scatter_add(2, newp2g, torch.ones(pred_logits.shape)).numpy()

    #pred_logits = pm/(vm+1e-12)
    #pred_logits =pred_logits[:,:,:2500]
    #gt_masks=gt_masks[:,:,:2500]

    aps=np.zeros((20,))
    for i in range(20):
        ap, temp = instance_segmentation_mAP(pred_logits>0.5, conf_logits, dataset, 0.05*(i+1))
        if i==0:
            ious= temp
        aps[i]=ap
        print('ap %d'%(i*5+5), ap)
        break
    print('mean ap', np.mean(aps))
    #return aps

    gen_visu(os.path.join(output_dir,vis_dir), dataset, pred_logits, conf_logits, ious)
    exit(0)


def merge_masks(masks, confs, finish):
    '''
    masks zoom_iteration n_shape x K x N
    conf zoom_iteration n_shape x K
    '''

    masks=[np.expand_dims(x,1) for x in masks]
    confs = [np.expand_dims(x,1) for x in confs]
    finish = [x for x in finish]
    masks = np.concatenate(masks, 1)
    confs=np.concatenate(confs,1)
    finish=np.concatenate(finish,1)

    n_shape, num_zoom_iteration, K, N = masks.shape
    all_ret=[]
    all_conf=[]
    for shape in range(n_shape):
        ret = []
        ret_confs = []
        for zoom_iteration in range(num_zoom_iteration):
            if finish[shape,zoom_iteration]<0.2:
                break
            cur_conf = confs[shape,zoom_iteration]
            cur_mask = masks[shape,zoom_iteration]
            m = np.max(cur_mask, 0)
            #cur_mask = np.logical_and(cur_mask >= (m), m>0)
            for i in range(K):
                if cur_conf[i]< 0.2:
                    continue
                m = cur_mask[i]
                m = m > 0.5
                if np.sum(m) == 0:
                    continue
                break_flag=False
                #for k in range(len(ret)):
                #    intersection = np.sum(np.logical_and(m, ret[k]))
                #    if intersection*10 >= np.sum(m) or intersection*10 >= np.sum(ret[k]):
                #        break_flag=True
                #        #ret[k] = np.logical_or(ret[k], m)
                #        if cur_conf[i] > ret_confs[k]:
                #            ret[k] = m
                #            ret_confs[k] = cur_conf[i]
                #        break
                if not break_flag:
                    ret.append(m)
                    ret_confs.append(cur_conf[i])
        while len(ret)> 0:
            merged_flag=False
            for i in range(len(ret)):
                break_flag=False
                for j in range(i):
                    intersection = np.sum(np.logical_and(ret[i], ret[j]))
                    if intersection*10>= np.sum(ret[i]) or intersection*10 >= np.sum(ret[j]):
                        ret[i] = np.logical_or(ret[i], ret[j])
                        ret_confs[i] = max(ret_confs[i], ret_confs[j])
                        del ret[j]
                        del ret_confs[j]
                        break_flag=True
                        merged_flag=True
                        break
                if break_flag:
                    break
            if not merged_flag:
                break
        print (len(ret))
        if len(ret)>0:
            ret = np.concatenate(ret, 0)
            ret = np.reshape(ret, (-1, N))
            ret_confs = np.array(ret_confs)
        else:
            ret = np.zeros((0, N))
            ret_confs = np.zeros((0, ))
        all_ret.append(ret)
        all_conf.append(ret_confs)
    size = max([x.shape[0] for x in all_conf])
    for i in range(len(all_ret)):
        temp = all_ret[i]
        all_ret[i] = np.concatenate([all_ret[i], np.zeros(( size-all_ret[i].shape[0], N))])
        all_conf[i] = np.concatenate([all_conf[i], np.zeros(( size-all_conf[i].shape[0], ))])

    all_conf = np.concatenate(all_conf, 0)
    all_ret = np.concatenate(all_ret, 0)
    all_ret = np.reshape( all_ret, (n_shape, size, N))
    all_conf = np.reshape( all_conf, (n_shape, size))
    return all_ret, all_conf

def instance_segmentation_mAP(pred_masks, confs, dataset, iou_threshold):
    '''
    pred_masks NUM_SHAPES x NUM_PRED_MASKS x N
    confs NUM_SHAPES x NUM_PRED_MASKS
    gt_masks NUM_SHAPES x NUM_GT_MASKS x N
    '''
    true_pos_list = []
    false_pos_list = []
    conf_score_list = []

    n_shape = pred_masks.shape[0]
    pred_n_ins = pred_masks.shape[1]
    ious = np.zeros((pred_masks.shape[0], pred_masks.shape[1]))
    gt_npos =0

    for i, data  in enumerate(dataset):
        cur_pred_mask = pred_masks[i]
        cur_pred_conf = confs[i]
        cur_gt_mask = data['ins_seg_label']
        gt_n_ins = cur_gt_mask.shape[0]
        gt_npos += np.sum(np.sum(cur_gt_mask, 1)>0)

        order =np.argsort(-cur_pred_conf)
        gt_used = np.zeros((gt_n_ins,), dtype=np.bool)
        gt_valid = np.sum(cur_gt_mask, 1) > 0

        for j in range(pred_n_ins):
            idx = order[j]

            if  cur_pred_conf[idx]> 0.2 and np.sum(cur_pred_mask[idx]) > 0:
                iou_max = 0.0; cor_gt_id = -1;
                for k in range(gt_n_ins):
                    if gt_valid[k] and (not gt_used[k]):
                        intersect = np.sum(np.logical_and(cur_gt_mask[k, :] , cur_pred_mask[idx, :]))
                        union = np.sum(np.logical_or(cur_gt_mask[k, :] , cur_pred_mask[idx, :]))
                        iou = intersect * 1.0 / union

                        if iou > iou_max:
                            iou_max = iou
                            cor_gt_id = k
                if iou_max > iou_threshold:
                    ious[i,idx] = iou_max
                    gt_used[cor_gt_id] = True

                    # add in a true positive
                    true_pos_list.append(True)
                    false_pos_list.append(False)
                    conf_score_list.append(cur_pred_conf[idx])

                else:
                    # add in a false positive
                    true_pos_list.append(False)
                    false_pos_list.append(True)
                    conf_score_list.append(cur_pred_conf[idx])

                #print(i,  cur_pred_conf[idx], iou_max)

    # compute AP
    true_pos = np.array(true_pos_list, dtype=np.float32)
    false_pos = np.array(false_pos_list, dtype=np.float32)
    conf_score = np.array(conf_score_list, dtype=np.float32)

    order = np.argsort(conf_score)
    sorted_true_pos = true_pos[order]
    sorted_false_pos = false_pos[order]

    ap = compute_ap(sorted_true_pos, sorted_false_pos, gt_npos)

    return ap, ious

def compute_ap(tp, fp, gt_npos, n_bins=100):
    assert len(tp) == len(fp), 'ERROR: the length of true_pos and false_pos is not the same!'

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    rec = tp / gt_npos
    prec = tp / (fp + tp)

    rec = np.insert(rec, 0, 0.0)
    prec = np.insert(prec, 0, 1.0)

    ap = 0.
    delta = 1.0 / n_bins

    out_rec = np.arange(0, 1 + delta, delta)
    out_prec = np.zeros((n_bins+1), dtype=np.float32)

    for idx, t in enumerate(out_rec):
        prec1 = prec[rec >= t]
        if len(prec1) == 0:
            p = 0.
        else:
            p = max(prec1)

        out_prec[idx] = p
        ap = ap + p / (n_bins + 1)
    return ap

def evaluate_foveal_segmentation(dataset,
                                viewed_masks,
                                proposal_logits,
                                finish_logits,
                                zoomed_points,
                               pred_logits,
                               conf_logits,
                               glob_pred_logits,
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

    seg_acc_per_class = defaultdict(float)
    num_inst_per_class = defaultdict(int)
    iou_per_class = defaultdict(float)

    all_ret, all_conf = merge_masks(glob_pred_logits, conf_logits, finish_logits)
    aps=np.zeros((20,))
    for i in range(20):
        ap, temp = instance_segmentation_mAP(all_ret>0.5, all_conf, dataset, 0.05*(i+1))
        if i==0:
            ious= temp
        aps[i]=ap
        print('ap %d'%(i*5+5), ap)
        break
    print('mean ap', np.mean(aps))
    #return aps
    gen_foveal_visu(os.path.join(output_dir,vis_dir), dataset, viewed_masks, proposal_logits, finish_logits,zoomed_points,  pred_logits, conf_logits, all_ret, all_conf, ious)
    exit(0)
