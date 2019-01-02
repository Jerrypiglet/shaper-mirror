import torch
from torch.utils.data import DataLoader

from shaper_compare.data.datasets import *
from shaper.data import transform as T
from shaper.utils.torch_util import set_random_seed


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
        # testing (might be different with training)
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

    if cfg.DATASET.TYPE == "ModelNetProto":
        if mode == "train":
            dataset = ModelNetCompare(root_dir=cfg.DATASET.ROOT_DIR,
                                      dataset_names=dataset_names,
                                      class_num_per_batch=cfg.DATASET.PROTO.CLASS_NUM_PER_BATCH,
                                      batch_support_num_per_class=cfg.DATASET.PROTO.BATCH_SUPPORT_NUM_PER_CLASS,
                                      batch_target_num=cfg.DATASET.PROTO.TRAIN_BATCH_TARGET_NUM,
                                      num_per_class=cfg.DATASET.PROTO.NUM_PER_CLASS,
                                      cross_num=cfg.DATASET.PROTO.CROSS_NUM,
                                      shuffle_data=is_train,
                                      shuffle_points=is_train,
                                      num_points=cfg.INPUT.NUM_POINTS,
                                      transform=transform,
                                      use_normal=cfg.INPUT.USE_NORMAL)
        elif mode == "val":
            dataset = ModelNetCompare(root_dir=cfg.DATASET.ROOT_DIR,
                                      dataset_names=dataset_names,
                                      class_num_per_batch=cfg.DATASET.PROTO.CLASS_NUM_PER_BATCH,
                                      batch_support_num_per_class=cfg.DATASET.PROTO.BATCH_SUPPORT_NUM_PER_CLASS,
                                      batch_target_num=cfg.DATASET.PROTO.VAL_BATCH_TARGET_NUM,
                                      num_per_class=cfg.DATASET.PROTO.NUM_PER_CLASS,
                                      cross_num=cfg.DATASET.PROTO.CROSS_NUM,
                                      shuffle_data=is_train,
                                      shuffle_points=is_train,
                                      num_points=cfg.INPUT.NUM_POINTS,
                                      transform=transform,
                                      use_normal=cfg.INPUT.USE_NORMAL)
        else:
            dataset = ModelNetCompare(root_dir=cfg.DATASET.ROOT_DIR,
                                      dataset_names=dataset_names,
                                      class_num_per_batch=cfg.DATASET.PROTO.CLASS_NUM_PER_BATCH,
                                      batch_support_num_per_class=cfg.DATASET.PROTO.BATCH_SUPPORT_NUM_PER_CLASS,
                                      batch_target_num=cfg.DATASET.PROTO.TEST_BATCH_TARGET_NUM,
                                      num_per_class=cfg.DATASET.PROTO.NUM_PER_CLASS,
                                      cross_num=cfg.DATASET.PROTO.CROSS_NUM,
                                      shuffle_data=is_train,
                                      shuffle_points=is_train,
                                      num_points=cfg.INPUT.NUM_POINTS,
                                      transform=transform,
                                      use_normal=cfg.INPUT.USE_NORMAL)
    elif cfg.DATASET.TYPE == "ShapeNet55Proto":
        if mode == "train":
            dataset = ShapeNet55Compare(root_dir=cfg.DATASET.ROOT_DIR,
                                        dataset_names=dataset_names,
                                        class_num_per_batch=cfg.DATASET.PROTO.CLASS_NUM_PER_BATCH,
                                        batch_support_num_per_class=cfg.DATASET.PROTO.BATCH_SUPPORT_NUM_PER_CLASS,
                                        batch_target_num=cfg.DATASET.PROTO.TRAIN_BATCH_TARGET_NUM,
                                        num_per_class=cfg.DATASET.PROTO.NUM_PER_CLASS,
                                        cross_num=cfg.DATASET.PROTO.CROSS_NUM,
                                        shuffle_data=is_train,
                                        shuffle_points=is_train,
                                        num_points=cfg.INPUT.NUM_POINTS,
                                        transform=transform,
                                        use_normal=cfg.INPUT.USE_NORMAL)
        elif mode == "val":
            dataset = ShapeNet55Compare(root_dir=cfg.DATASET.ROOT_DIR,
                                        dataset_names=dataset_names,
                                        class_num_per_batch=cfg.DATASET.PROTO.CLASS_NUM_PER_BATCH,
                                        batch_support_num_per_class=cfg.DATASET.PROTO.BATCH_SUPPORT_NUM_PER_CLASS,
                                        batch_target_num=cfg.DATASET.PROTO.VAL_BATCH_TARGET_NUM,
                                        num_per_class=cfg.DATASET.PROTO.NUM_PER_CLASS,
                                        cross_num=cfg.DATASET.PROTO.CROSS_NUM,
                                        shuffle_data=is_train,
                                        shuffle_points=is_train,
                                        num_points=cfg.INPUT.NUM_POINTS,
                                        transform=transform,
                                        use_normal=cfg.INPUT.USE_NORMAL)
        else:
            dataset = ShapeNet55Compare(root_dir=cfg.DATASET.ROOT_DIR,
                                        dataset_names=dataset_names,
                                        class_num_per_batch=cfg.DATASET.PROTO.CLASS_NUM_PER_BATCH,
                                        batch_support_num_per_class=cfg.DATASET.PROTO.BATCH_SUPPORT_NUM_PER_CLASS,
                                        batch_target_num=cfg.DATASET.PROTO.TEST_BATCH_TARGET_NUM,
                                        num_per_class=cfg.DATASET.PROTO.NUM_PER_CLASS,
                                        cross_num=cfg.DATASET.PROTO.CROSS_NUM,
                                        shuffle_data=is_train,
                                        shuffle_points=is_train,
                                        num_points=cfg.INPUT.NUM_POINTS,
                                        transform=transform,
                                        use_normal=cfg.INPUT.USE_NORMAL)

    else:
        raise ValueError("Undefined dataset type: {}.".format(cfg.DATASET.TYPE))

    return dataset


def build_dataloader(cfg, mode="train"):
    assert mode in ["train", "val", "test"]
    # if mode == "train":
    #     batch_size = cfg.TRAIN.BATCH_SIZE
    # else:
    #     batch_size = cfg.TEST.BATCH_SIZE
    if cfg.DATALOADER.RNG_SEED >= 0:
        set_random_seed(cfg.DATALOADER.RNG_SEED)
    dataset = build_dataset(cfg, mode)
    data_loader = DataLoader(
        dataset,
        batch_size=dataset.get_batch_size(),
        shuffle=False,  # shuffle is done in the dataset
        drop_last=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )
    return data_loader


if __name__ == "__main__":
    from shaper_proto.config import cfg

    cfg.merge_from_file(
        "/home/rayc/Projects/shaper/configs/modelnet40/proto_net/pn2ssg/pn2ssg_source_cls_without_dir.yaml")

    cfg.freeze()

    data_loader = build_dataloader(cfg)

    for iteration, data_batch in enumerate(data_loader):

        for k, v in data_batch.items():
            print("{}: {}".format(k, v))

        break
