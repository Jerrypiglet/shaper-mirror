"""Generalized tester"""

import logging
import time
from collections import defaultdict

import numpy as np

import torch
from torch import nn

from shaper.models import build_model
from shaper.data import build_dataloader
from shaper.utils.checkpoint import Checkpointer
from shaper.utils.metric_logger import MetricLogger


def test_model(model,
               loss_fn,
               metric_fn,
               data_loader,
               log_period=1,
               with_label=True):
    """Test model

    In some case, the model is tested without labels, where loss_fn and metric_fn are invalid.
    This method will forward the model to get predictions in the order of dataloader.

    Notes:
        This method will store all the prediction, which might consume large memory.

    Args:
        model (nn.Module): model to test
        loss_fn (nn.Module or Function): loss function
        metric_fn (nn.Module or Function): metric function
        data_loader (torch.utils.data.DataLoader):
        log_period (int):
        with_label (bool): whether dataloader has labels

    Returns:
        meters (MetricLogger)
        test_result_dict (dict)

    """
    logger = logging.getLogger("shaper.test")
    meters = MetricLogger(delimiter="  ")
    loss_fn.eval()
    model.eval()
    metric_fn.eval()

    test_result_dict = defaultdict(list)

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

    return meters, test_result_dict


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

    # test
    start_time = time.time()
    test_meters, test_result_dict = test_model(model,
                                               loss_fn,
                                               metric_fn,
                                               test_data_loader,
                                               log_period=cfg.TEST.LOG_PERIOD)
    test_time = time.time() - start_time
    logger.info("Test {}  forward time: {:.2f}s".format(test_meters.summary_str, test_time))
