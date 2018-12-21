import logging
import time
from collections import defaultdict

import numpy as np
import scipy.stats as stats

import torch
from torch import nn

from shaper_compare.models import build_model
from shaper_compare.data import build_dataloader
from shaper.data.build import build_transform
from shaper_compare.data.datasets import evaluate_classification
from shaper_compare.utils.checkpoint import Checkpointer
from shaper.utils.metric_logger import MetricLogger
from shaper.utils.io import mkdir


def test_model(model,
               loss_fn,
               metric_fn,
               data_loader,
               log_period=1,
               with_label=True):
    """Test model

    In some case, the model is tested without labels, where loss_fn and metric_fn are invalid.
    This method will forward the model to get predictions in the order of dataloader.

    Args:
        model (nn.Module): model to test
        loss_fn (nn.Module or Function): loss function
        metric_fn (nn.Module or Function): metric function
        data_loader (torch.utils.data.DataLoader):
        log_period (int):
        with_label (bool): whether dataloader has labels

    Returns:

    """
    logger = logging.getLogger("shaper.test")
    meters = MetricLogger(delimiter="  ")
    model.eval()

    test_result_dict = defaultdict(list)
    metric_result_dict = defaultdict(list)
    data_loader.dataset.reset()

    with torch.no_grad():
        end = time.time()
        for iteration, data_batch in enumerate(data_loader):
            data_time = time.time() - end

            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

            preds = model(data_batch)

            for k, v in preds.items():
                test_result_dict[k].append(v.cpu().numpy())

            if with_label:
                loss_dict = loss_fn(preds, data_batch)
                metric_dict = metric_fn(preds, data_batch)

                for k, v in metric_dict.items():
                    metric_result_dict[k].append(v.cpu().numpy())

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

    # concatenate
    test_result_dict = {k: np.concatenate(v, axis=0) for k, v in test_result_dict.items()}
    metric_result_dict = {k: np.concatenate(v, axis=0) for k, v in metric_result_dict.items()}
    return meters, test_result_dict, metric_result_dict


def test(cfg, output_dir=""):
    logger = logging.getLogger("shaper.tester")

    # build model
    model, loss_fn, metric_fn = build_model(cfg)
    model = nn.DataParallel(model).cuda()

    # build checkpointer
    checkpointer = Checkpointer(model, save_dir=output_dir)

    if cfg.TEST.WEIGHT:
        weight_path = cfg.TEST.WEIGHT.replace("@", output_dir)
        checkpointer.load(weight_path, resume=False)
    else:
        checkpointer.load(None, resume=True)

    # build data loader
    test_data_loader = build_dataloader(cfg, mode="test")
    test_dataset = test_data_loader.dataset
    logger.info("Dataset MD5: {}".format(test_dataset.get_md5_list()))

    # test
    test_result_collection = []
    metric_result_collection = []
    vis_dir = cfg.TEST.VIS_DIR.replace("@", output_dir)
    if vis_dir:
        mkdir(vis_dir)

    if cfg.TEST.VOTE.ENABLE:  # Multi-view voting
        raise NotImplementedError
        for view_ind in range(cfg.TEST.VOTE.NUM_VIEW):
            start_time = time.time()
            tmp_cfg = cfg.clone()
            tmp_cfg.defrost()
            angle = 2 * np.pi * view_ind / cfg.TEST.VOTE.NUM_VIEW
            tmp_cfg.TEST.AUGMENTATION = (("PointCloudRotateByAngle", cfg.TEST.VOTE.AXIS, angle),)
            test_data_loader.dataset.transform = build_transform(tmp_cfg, False)
            test_meters, test_result_dict, metric_result_dict = test_model(model,
                                                                           loss_fn,
                                                                           metric_fn,
                                                                           test_data_loader,
                                                                           log_period=cfg.TEST.LOG_PERIOD)

            test_result_collection.append(test_result_dict)
            test_time = time.time() - start_time
            logger.info("Test rotation over [{}] axis by [{:.4f}] rad".format(cfg.TEST.VOTE.AXIS, angle))
            logger.info("Test {}  forward time: {:.2f}s".format(test_meters.summary_str, test_time))
    else:
        start_time = time.time()
        test_meters, test_result_dict, metric_result_dict = test_model(model,
                                                                       loss_fn,
                                                                       metric_fn,
                                                                       test_data_loader,
                                                                       log_period=cfg.TEST.LOG_PERIOD)
        test_result_collection.append(test_result_dict)
        metric_result_collection.append(metric_result_dict)
        test_time = time.time() - start_time
        logger.info("Test {}  forward time: {:.2f}s".format(test_meters.summary_str, test_time))

    # ---------------------------------------------------------------------------- #
    # Ensemble
    # ---------------------------------------------------------------------------- #
    # For classification, only use 'relation_preds_per_class'
    cls_preds_all = [d["relation_preds_per_class"] for d in test_result_collection]
    # sanity check
    assert all(len(cls_logits) == test_dataset.get_total_target_num() for cls_logits in cls_preds_all)
    # remove transform
    test_dataset.transform = None

    if cfg.TEST.VOTE.ENABLE:
        raise NotImplementedError
        for score_heur in cfg.TEST.VOTE.SCORE_HEUR:
            if score_heur == "soft_label":
                cls_logits_ensemble = np.mean(cls_preds_all, axis=0)
                pred_labels = np.argmax(cls_logits_ensemble, -1)  # (num_samples,)
            elif score_heur == "label":
                pred_labels_all = np.argmax(cls_preds_all, -1)
                pred_labels = stats.mode(pred_labels_all, axis=0)[0].squeeze(0)
            else:
                raise ValueError("Unknown score heuristic")

            logger.info("Ensemble using [{}] with [{}] rotations over [{}] axis.".format(
                score_heur, cfg.TEST.VOTE.NUM_VIEW, cfg.TEST.VOTE.AXIS))

            overall_acc, acc_per_class = evaluate_classification(test_dataset, pred_labels,
                                                                 output_dir=output_dir,
                                                                 vis_dir=vis_dir,
                                                                 suffix=score_heur)

    else:
        pred_labels = np.argmax(cls_preds_all[0], -1)
        overall_acc, acc_per_class = evaluate_classification(test_dataset, pred_labels,
                                                             aux_preds=test_result_collection[0],
                                                             output_dir=output_dir,
                                                             vis_dir=vis_dir)
        total_correct_num = np.sum(metric_result_collection[0]["acc"])
        metric_accuracy = total_correct_num / pred_labels.shape[0]
        print("pred accuracy: {:.4f}".format(overall_acc))
        print("metric accuracy {:.4f}".format(metric_accuracy))

    return test_dataset, cls_preds_all, overall_acc, acc_per_class
