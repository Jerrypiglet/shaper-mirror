import torch
from torch.utils.data import DataLoader

from shaper.data.build import build_transform
from shaper.utils.torch_util import set_random_seed
from . import datasets as D


def build_dataset(cfg, mode="train"):
    if mode == "train":
        dataset_names = cfg.DATASET.TRAIN
    elif mode == "val":
        dataset_names = cfg.DATASET.VAL
    else:
        dataset_names = cfg.DATASET.TEST

    is_train = mode == "train"
    transform = build_transform(cfg, is_train)

    # Notice that shuffle_points is enabled for few shot training.
    if cfg.DATASET.TYPE == "MODELNET_FEWSHOT":
        dataset = D.ModelNetFewShot(root_dir=cfg.DATASET.ROOT_DIR,
                                    dataset_names=dataset_names,
                                    k_shot=cfg.DATASET.FEWSHOT.K,
                                    cross_index=cfg.DATASET.FEWSHOT.CROSS_INDEX,
                                    shuffle_points=is_train,
                                    num_points=cfg.INPUT.NUM_POINTS,
                                    transform=transform,
                                    use_normal=cfg.INPUT.USE_NORMAL)
    elif cfg.DATASET.TYPE == "SHAPENET_FEWSHOT":
        dataset = D.ShapeNetFewShot(root_dir=cfg.DATASET.ROOT_DIR,
                                    dataset_names=dataset_names,
                                    k_shot=cfg.DATASET.FEWSHOT.K,
                                    cross_index=cfg.DATASET.FEWSHOT.CROSS_INDEX,
                                    shuffle_points=is_train,
                                    num_points=cfg.INPUT.NUM_POINTS,
                                    transform=transform,
                                    )
    else:
        raise NotImplementedError()

    return dataset


def build_dataloader(cfg, mode="train"):
    assert mode in ["train", "val", "test"]
    if mode == "train":
        batch_size = cfg.TRAIN.BATCH_SIZE
    else:
        batch_size = cfg.TEST.BATCH_SIZE

    if cfg.DATALOADER.RNG_SEED >= 0:
        set_random_seed(cfg.DATALOADER.RNG_SEED)

    is_train = mode == "train"
    dataset = build_dataset(cfg, mode)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        # shuffle=False,
        drop_last=False,  # set False for fewshot learning
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )
    return data_loader
