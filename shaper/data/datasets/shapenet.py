import os.path as osp
import json

import numpy as np

import torch
from torch.utils.data import Dataset


class ShapeNet(Dataset):
    ROOT_DIR = "data/shapenet"
    cat_file = "synsetoffset2category.txt"
    split_dir = "train_test_split"
    dataset_map = {
        "train": "shuffled_train_file_list.json",
        "val": "shuffled_val_file_list.json",
        "test": "shuffled_test_file_list.json",
    }

    def __init__(self, root_dir, dataset_names, transform=None,
                 num_points=-1, sample_points=True):
        self.root_dir = root_dir
        self.datasets_names = dataset_names
        self.num_points = num_points
        self.sample_points = sample_points
        self.transform = transform

        # classes
        self.class_to_offset_map = self._load_cat_file()
        self.offset_to_class_map = {v: k for k, v in self.class_to_offset_map.items()}
        self.classes = list(self.class_to_offset_map.keys())
        sorted(self.classes)
        self.classes_to_ind_map = {c: i for i, c in enumerate(self.classes)}

        # meta data
        self.meta_data = []
        for dataset_name in dataset_names:
            meta_data = self._load_dataset(dataset_name)
            self.meta_data.extend(meta_data)
        print("{} classes with {} models".format(len(self.classes), len(self.meta_data)))

        # TODO: support preload

    def _load_cat_file(self):
        class_to_offset_map = {}
        with open(osp.join(self.root_dir, self.cat_file), 'r') as fid:
            for line in fid:
                class_name, class_dir = line.strip().split()
                class_to_offset_map[class_name] = class_dir
        return class_to_offset_map

    def _load_dataset(self, dataset_name):
        split_fname = osp.join(self.root_dir, self.split_dir, self.dataset_map[dataset_name])
        fname_list = json.load(open(split_fname, 'r'))
        meta_data = []
        for fname in fname_list:
            # fname = fname.replace("shape_data", self.root_dir)
            _, offset, token = fname.split("/")
            pts_path = osp.join(self.root_dir, offset, "points", token + '.pts')
            class_name = self.offset_to_class_map[offset]
            meta_data.append({'token': token,
                              'class': class_name,
                              'pts': pts_path,
                              })
        return meta_data

    def __getitem__(self, index):
        meta_data = self.meta_data[index]
        class_name = meta_data["class"]
        class_ind = self.classes_to_ind_map[class_name]
        pts_path = meta_data["pts"]
        points = np.loadtxt(pts_path).astype(np.float32)
        # print(index, class_name, class_ind, pts_path)

        if self.num_points > 0:
            choice = np.random.choice(len(points), self.num_points, replace=not self.sample_points)
        else:
            choice = np.arange(len(points))
        points = points[choice]

        # points = points.transpose()  # [in_channels, num_points]

        if self.transform is not None:
            points = self.transform(points)

        return {
            "points": points,
            "cls_labels": class_ind,
        }

    def __len__(self):
        return len(self.meta_data)
