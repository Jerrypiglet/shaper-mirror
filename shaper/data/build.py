import torch

from .datasets import *
from . import transform as T


def build_transform(cfg, is_train=True):
    if is_train:
        transform_list = [T.PointCloudToTensor()]
        for aug in cfg.TRAIN.AUGMENTATION:
            if isinstance(aug, (list, tuple)):
                transform_list.append(getattr(T, aug[0])(*aug[1:]))
            else:
                transform_list.append(getattr(T, aug)())
        transform_list.append(T.PointCloudTensorTranspose())
        transform = T.Compose(transform_list)
    else:
        # testing
        transform_list = [T.PointCloudToTensor()]
        for aug in cfg.TEST.AUGMENTATION:
            if isinstance(aug, (list, tuple)):
                transform_list.append(getattr(T, aug[0])(*aug[1:]))
            else:
                transform_list.append(getattr(T, aug)())
        transform_list.append(T.PointCloudTensorTranspose())
        transform = T.Compose(transform_list)


    return transform


def build_dataset(cfg, mode="train"):
    if mode == "train":
        dataset_names = cfg.DATASET.TRAIN
    elif mode == "val":
        dataset_names = cfg.DATASET.VAL
    else:
        dataset_names = cfg.DATASET.TEST

    is_train = mode == "train"
    transform = build_transform(cfg, is_train)

    if cfg.DATASET.TYPE == "ShapeNet":
        dataset = ShapeNet(root_dir=cfg.DATASET.ROOT_DIR,
                           dataset_names=dataset_names,
                           shuffle_points=is_train,
                           num_points=cfg.INPUT.NUM_POINTS,
                           transform=transform)
    elif cfg.DATASET.TYPE == "ModelNet":
        dataset = ModelNet(root_dir=cfg.DATASET.ROOT_DIR,
                           dataset_names=dataset_names,
                           shuffle_points=False, num_points=cfg.INPUT.NUM_POINTS,
                           transform=transform)
    else:
        raise NotImplementedError()

    return dataset


def build_dataloader(cfg, mode="train"):
    assert mode in ["train", "val", "test"]
    if mode == "train":
        batch_size = cfg.TRAIN.BATCH_SIZE
    else:
        batch_size = cfg.TEST.BATCH_SIZE

    is_train = mode == "train"
    dataset = build_dataset(cfg, mode)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=cfg.DATALOADER.NUM_WORKERS)
    return data_loader
