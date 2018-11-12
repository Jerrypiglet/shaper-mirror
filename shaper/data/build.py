import torch

from .datasets import *
from . import transform as T


def build_transform(cfg, is_train=True):
    if is_train:
        transform_list = []
        for aug in cfg.TRAIN.AUGMENTATION:
            transform_list.append(getattr(T, aug[0])(*aug[1:]))
        transform_list.append(T.PointCloudToTensor())
        transform = T.Compose(transform_list)
    else:
        transform = T.PointCloudToTensor()
    return transform


def build_dataset(cfg, is_train=True):
    if is_train:
        dataset_names = cfg.DATASET.TRAIN
    else:
        dataset_names = cfg.DATASET.TEST

    transform = build_transform(cfg, is_train)

    if cfg.DATASET.TYPE == "ShapeNet":
        dataset = ShapeNet(root_dir=cfg.DATASET.ROOT_DIR,
                           dataset_names=dataset_names,
                           sample_points=False,
                           num_points=cfg.INPUT.NUM_POINTS,
                           transform=transform)
    elif cfg.DATASET.TYPE == "ModelNet":
        dataset = ModelNet(root_dir=cfg.DATASET.ROOT_DIR,
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
