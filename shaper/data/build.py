import torch

from .datasets import *


def build_dataset(cfg, is_train=True):
    if is_train:
        dataset_names = cfg.DATASET.TRAIN
    else:
        dataset_names = cfg.DATASET.TEST

    if cfg.DATASET.TYPE == "ShapeNet":
        dataset = ShapeNet(root_dir=ShapeNet.ROOT_DIR,
                           dataset_names=dataset_names,
                           shuffle_points=True, num_points=1024)
    elif cfg.DATASET.TYPE == "ModelNet":
        dataset = ModelNet(root_dir=ModelNet.ROOT_DIR,
                           dataset_names=dataset_names,
                           shuffle_points=True, num_points=1024)
    else:
        raise NotImplementedError()

    return dataset


def build_dataloader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.TRAIN.BATCH_SIZE
    else:
        batch_size = cfg.TEST.BATCH_SIZE

    dataset = build_dataset(cfg, is_train)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=cfg.DATALOADER.NUM_WORKERS)
    return data_loader
