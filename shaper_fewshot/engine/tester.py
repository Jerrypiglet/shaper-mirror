import logging
import time
import os.path as osp

import numpy as np
import scipy.stats as stats

import torch
from torch import nn

from shaper_fewshot.models import build_model
from shaper_fewshot.data import build_dataloader
from shaper.data.build import build_transform
from shaper.data.datasets import evaluate_classification
from shaper_fewshot.utils.checkpoint import CheckpointerFewshot
from shaper.utils.io import mkdir, write_pkl
from shaper.utils.np_util import np_softmax
from shaper.engine.tester import test_model


def test(cfg, output_dir="",
         save_pred=False,
         save_eval=False,
         suffix=""):
    logger = logging.getLogger("shaper.tester")

    # build model
    model, loss_fn, metric_fn = build_model(cfg)
    model = nn.DataParallel(model).cuda()

    # build checkpointer
    checkpointer = CheckpointerFewshot(model, save_dir=output_dir, logger=logger)

    if cfg.TEST.WEIGHT:
        weight_path = cfg.TEST.WEIGHT.replace("@", output_dir)
        checkpointer.load(weight_path, resume=False, pretrained=False)
    else:
        checkpointer.load(None, resume=True, pretrained=False)

    # build data loader
    test_data_loader = build_dataloader(cfg, mode="test")
    test_dataset = test_data_loader.dataset

    # test
    test_result_collection = []
    vis_dir = cfg.TEST.VIS_DIR.replace("@", output_dir)
    if vis_dir:
        mkdir(vis_dir)
    if suffix:
        suffix = '_' + suffix

    if cfg.TEST.VOTE.ENABLE:  # Multi-view voting
        for view_ind in range(cfg.TEST.VOTE.NUM_VIEW):
            start_time = time.time()
            tmp_cfg = cfg.clone()
            tmp_cfg.defrost()
            angle = 2 * np.pi * view_ind / cfg.TEST.VOTE.NUM_VIEW
            tmp_cfg.TEST.AUGMENTATION = (("PointCloudRotateByAngle", cfg.TEST.VOTE.AXIS, angle),)
            test_data_loader.dataset.transform = build_transform(tmp_cfg, False)
            test_meters, test_result_dict = test_model(model,
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
        test_meters, test_result_dict = test_model(model,
                                                   loss_fn,
                                                   metric_fn,
                                                   test_data_loader,
                                                   log_period=cfg.TEST.LOG_PERIOD)
        test_result_collection.append(test_result_dict)
        test_time = time.time() - start_time
        logger.info("Test {}  forward time: {:.2f}s".format(test_meters.summary_str, test_time))

    # ---------------------------------------------------------------------------- #
    # Ensemble
    # ---------------------------------------------------------------------------- #
    # For classification, only use 'cls_logit'
    cls_logit_collection = [d["cls_logit"] for d in test_result_collection]
    # sanity check
    assert all(len(cls_logit) == len(test_dataset) for cls_logit in cls_logit_collection)

    # Save predictions
    if save_pred:
        pred_fname = osp.join(output_dir, "pred" + suffix + ".pkl")
        write_pkl(test_result_collection, pred_fname)
        logger.info("Write predictions into {:s}.".format(pred_fname))


    if cfg.TEST.VOTE.ENABLE:
        for score_heur in cfg.TEST.VOTE.SCORE_HEUR:
            if score_heur == "logit":
                cls_logit_ensemble = np.mean(cls_logit_collection, axis=0)
                pred_labels = np.argmax(cls_logit_ensemble, -1)  # (num_samples,)
            elif score_heur == "softmax":
                cls_prob_collection = np_softmax(np.asarray(cls_logit_collection))
                cls_prob_ensemble = np.mean(cls_prob_collection, axis=0)
                pred_labels = np.argmax(cls_prob_ensemble, -1)
            elif score_heur == "label":
                pred_label_collection = np.argmax(cls_logit_collection, -1)
                pred_labels = stats.mode(pred_label_collection, axis=0)[0].squeeze(0)
            else:
                raise ValueError("Unknown score heuristic")

            logger.info("Ensemble using [{}] with [{}] rotations over [{}] axis.".format(
                score_heur, cfg.TEST.VOTE.NUM_VIEW, cfg.TEST.VOTE.AXIS))

            eval_results = evaluate_classification(test_dataset, pred_labels,
                                                   output_dir=output_dir,
                                                   vis_dir=vis_dir,
                                                   suffix=score_heur)

    else:
        pred_labels = np.argmax(cls_logit_collection[0], -1)
        eval_results = evaluate_classification(test_dataset, pred_labels,
                                               aux_preds=test_result_collection[0],
                                               output_dir=output_dir,
                                               vis_dir=vis_dir)

    # Save eval result
    if save_eval:
        eval_result_fname = osp.join(output_dir, "eval" + suffix + ".pkl")
        write_pkl(eval_results, eval_result_fname)
        logger.info("Write eval results into {:s}.".format(eval_result_fname))
