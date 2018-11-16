import logging
import time
import numpy as np
import os.path as osp
import imageio

import torch
from torch import nn

from shaper.models import build_model
from shaper.data import build_dataloader
from shaper.data.build import build_transform
from shaper.utils.torch_utils import set_random_seed
from shaper.utils.checkpoint import Checkpointer
from shaper.utils.metric_logger import MetricLogger
from shaper.utils.np_utils import softmax
from shaper.utils.pc_util import point_cloud_three_views
from shaper.utils.io import mkdir


def cal_avg_class_acc(pred, label, num_classes, visual=False, prefix="",
                      points=None, shape_names=None, pic_path=""):
    total_seen_class = [0 for _ in range(num_classes)]
    total_correct_class = [0 for _ in range(num_classes)]
    err_cnt = 0
    assert (not visual) or ((points is not None) and (shape_names is not None) or (pic_path != ""))
    if not osp.exists(pic_path):
        mkdir(pic_path)
    for i in range(pred.shape[0]):
        l = label[i]
        total_seen_class[l] += 1
        total_correct_class[l] += (pred[i] == l)

        if pred[i] != l and visual:
            if prefix == "":
                file_name = "{:04d}_label_{}_pred_{}".format(
                    err_cnt, shape_names[l], shape_names[pred[i]])
                img_file_name = file_name + ".jpg"
                xyz_file_name = file_name + ".xyz"
            else:
                file_name = prefix+"_{:04d}_label_{}_pred_{}".format(
                    err_cnt, shape_names[l], shape_names[pred[i]])
                img_file_name = file_name + ".jpg"
                xyz_file_name = file_name + ".xyz"

            img_file_name = osp.join(pic_path, img_file_name)
            xyz_file_name = osp.join(pic_path, xyz_file_name)

            wrong_pts = np.squeeze(points[i, ...])
            wrong_pts = np.transpose(wrong_pts)
            out_img = point_cloud_three_views(wrong_pts)
            imageio.imwrite(img_file_name, out_img)

            np.savetxt(xyz_file_name, wrong_pts, fmt="%.4f")

            err_cnt += 1

    avg_class_acc = np.mean(np.array(total_correct_class)/np.array(total_seen_class, dtype=np.float))

    return avg_class_acc






def test_model(model,
               loss_fn,
               metric_fn,
               data_loader,
               log_period=1):
    logger = logging.getLogger("shaper.test")
    meters = MetricLogger(delimiter="  ")
    model.eval()
    end = time.time()

    test_result_dict = {
        "points": [],
        "cls_labels": [],
        "cls_logits": [],
    }

    with torch.no_grad():
        for iteration, data_batch in enumerate(data_loader):
            data_time = time.time() - end

            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

            preds = model(data_batch)

            test_result_dict["points"].append(data_batch["points"])
            test_result_dict["cls_labels"].append(data_batch["cls_labels"])
            test_result_dict["cls_logits"].append(preds["cls_logits"])

            loss_dict = loss_fn(preds, data_batch)
            # for key, value in loss_dict.items():
            #     if key not in test_result_dict.keys():
            #         test_result_dict[key] = []
            #     test_result_dict[key].append(value)

            metric_dict = metric_fn(preds, data_batch)
            for key, value in metric_dict.items():
                if key not in test_result_dict.keys():
                    test_result_dict[key] = []
                test_result_dict[key].append(value)

            losses = sum(loss_dict.values())
            meters.update(loss=losses, **loss_dict, **metric_dict)
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            if iteration % log_period == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "iter: {iter:4d}",
                            "{meters}",
                        ]
                    ).format(
                        iter=iteration,
                        meters=str(meters),
                    )
                )

    # transform to np.array and concatenate
    for key, value_list in test_result_dict.items():
        val_numpy = []
        if isinstance(value_list, (list, tuple)):
            for value in value_list:
                _ = value.cpu().numpy()
                val_numpy.append(_)
            val_numpy = np.concatenate(val_numpy, axis=0)
            test_result_dict[key] = val_numpy

    return meters, test_result_dict


def test(cfg, output_dir=""):
    set_random_seed(cfg.RNG_SEED)
    logger = logging.getLogger("shaper.tester")

    # build model
    model, loss_fn, metric_fn = build_model(cfg)
    device_ids = cfg.DEVICE_IDS if cfg.DEVICE_IDS else None
    model = nn.DataParallel(model, device_ids=device_ids).cuda()

    # build checkpointer
    checkpointer = Checkpointer(model, save_dir=output_dir)

    if cfg.TEST.TEST_BEST:
        checkpoint_data = checkpointer.load("model_best", resume=False)
    else:
        checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, resume=True)

    # build data loader
    test_data_loader = build_dataloader(cfg, mode="test")

    # visualization
    visual = cfg.TEST.VISUAL
    shape_names = None
    if visual:
        assert osp.exists(cfg.DATASET.SHAPE_NAME_PATH)
        shape_names = [line.rstrip() for line in \
                   open(cfg.DATASET.SHAPE_NAME_PATH)]

    # test
    test_result_collection = {}

    start_time = time.time()
    assert cfg.TEST.TYPE in ["Vanilla", "Vote"]
    if cfg.TEST.TYPE == "Vanilla":
        test_meters, test_result_dict = test_model(model,
                                                   loss_fn,
                                                   metric_fn,
                                                   test_data_loader, log_period=cfg.TEST.LOG_PERIOD)
        test_result_collection.update(test_result_dict)
    elif cfg.TEST.TYPE == "Vote":
        for i in range(cfg.TEST.VOTE.NUMBER):
            period_start_time = time.time()
            test_result_collection[i] = {}
            cfg.defrost()
            angle = i / cfg.TEST.VOTE.NUMBER * 2 * np.pi
            test_result_collection[i]["angle"] = angle
            cfg.TEST.AUGMENTATION = (("PointCloudRotateByAngle", cfg.TEST.VOTE.AXIS, angle),)
            test_data_loader.dataset.transform = build_transform(cfg, False)
            cfg.freeze()
            test_meters, test_result_dict = test_model(model,
                                                       loss_fn,
                                                       metric_fn,
                                                       test_data_loader, log_period=cfg.TEST.LOG_PERIOD)
            test_result_collection[i].update(test_result_dict)
            period_test_time = time.time() - period_start_time
            logger.info("Test rotation over [{}] by [{:.4f}] rad".format(cfg.TEST.VOTE.AXIS, angle))
            logger.info("Test {}  period_time: {:.2f}s".format(test_meters.summary_str, period_test_time))

        # Vote
        ensemble_sum_logits = None
        ensemble_sum_softmax = None
        ensemble_sum_label = None
        cls_labels = None
        num_classes = None
        points = None

        for index, test_result_dict in test_result_collection.items():
            if cls_labels is None:
                cls_labels = test_result_dict['cls_labels']
            if num_classes is None:
                num_classes = test_result_dict['cls_logits'].shape[1]
            if visual and (points is None):
                points = test_result_dict['points']

            assert set(cfg.TEST.VOTE.TYPE) <= set(["Logits", "Softmax", "Label"])
            if "Logits" in cfg.TEST.VOTE.TYPE:
                if ensemble_sum_logits is None:
                    ensemble_sum_logits = np.zeros(test_result_dict['cls_logits'].shape)
                ensemble_sum_logits += test_result_dict['cls_logits']
            if "Softmax" in cfg.TEST.VOTE.TYPE:
                if ensemble_sum_softmax is None:
                    ensemble_sum_softmax = np.zeros(test_result_dict['cls_logits'].shape)
                ensemble_sum_softmax += softmax(test_result_dict['cls_logits'])
            if "Label" in cfg.TEST.VOTE.TYPE:
                if ensemble_sum_label is None:
                    ensemble_sum_label = np.zeros(test_result_dict['cls_logits'].shape)
                max_inds = np.argmax(test_result_dict['cls_logits'], axis=-1)

                one_hot_pred = np.zeros(test_result_dict['cls_logits'].shape)
                for row_index in range(one_hot_pred.shape[0]):
                    one_hot_pred[row_index, max_inds[row_index]] = 1.0
                ensemble_sum_label += one_hot_pred
        logger.info("Ensemble [{}] rotations over [{}] axis: ".format(cfg.TEST.VOTE.NUMBER, cfg.TEST.VOTE.AXIS))

        if "Logits" in cfg.TEST.VOTE.TYPE:
            ensemble_pred_logits = np.argmax(ensemble_sum_logits, -1)
            accuracy_logits = np.mean(ensemble_pred_logits == cls_labels)
            if visual:
                avg_class_acc = cal_avg_class_acc(ensemble_pred_logits, cls_labels, num_classes, visual=True,
                                                  prefix="logits", points=points, shape_names=shape_names,
                                                  pic_path=cfg.TEST.PIC_PATH)
            else:
                avg_class_acc = cal_avg_class_acc(ensemble_pred_logits, cls_labels, num_classes)
            logger.info("Ensemble logits  pred accuracy: {:.4f}  avg class accuracy: {:.4f}".format(
                accuracy_logits, avg_class_acc))

        if "Softmax" in cfg.TEST.VOTE.TYPE:
            ensemble_pred_softmax = np.argmax(ensemble_sum_softmax, -1)
            accuracy_softmax = np.mean(ensemble_pred_softmax == cls_labels)
            if visual:
                avg_class_acc = cal_avg_class_acc(ensemble_pred_softmax, cls_labels, num_classes, visual=True,
                                                  prefix="softmax", points=points, shape_names=shape_names,
                                                  pic_path=cfg.TEST.PIC_PATH)
            else:
                avg_class_acc = cal_avg_class_acc(ensemble_pred_softmax, cls_labels, num_classes)
            logger.info("Ensemble softmax pred accuracy: {:.4f}  avg class accuracy: {:.4f}".format(
                accuracy_softmax, avg_class_acc))

        if "Label" in cfg.TEST.VOTE.TYPE:
            ensemble_pred_label = np.argmax(ensemble_sum_label, -1)
            accuracy_label = np.mean(ensemble_pred_label == cls_labels)
            if visual:
                avg_class_acc = cal_avg_class_acc(ensemble_pred_label, cls_labels, num_classes, visual=True,
                                                  prefix="label", points=points, shape_names=shape_names,
                                                  pic_path=cfg.TEST.PIC_PATH)
            else:
                avg_class_acc = cal_avg_class_acc(ensemble_pred_label, cls_labels, num_classes)
            logger.info("Ensemble label   pred accuracy: {:.4f}  avg class accuracy: {:.4f}".format(
                accuracy_label, avg_class_acc))

    total_test_time = time.time() - start_time
    logger.info("Test finish  total_time: {:.2f}s".format(total_test_time))
    return model
