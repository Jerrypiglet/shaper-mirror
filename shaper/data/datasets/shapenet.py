import os
import os.path as osp
import json

import numpy as np

import torch
from torch.utils.data import Dataset
# from utils.open3d_visualize import Visualizer


class ShapeNet(Dataset):
    ROOT_DIR = "data/shapenet"
    cat_file = "synsetoffset2category.txt"
    split_dir = "train_test_split"
    dataset_map = {
        "train": "shuffled_train_file_list.json",
        "val": "shuffled_val_file_list.json",
        "test": "shuffled_test_file_list.json",
    }

    def __init__(self, root_dir, dataset_names,
                 shuffle_points=False, num_points=-1):
        self.root_dir = root_dir
        self.datasets_names = dataset_names
        self.num_points = num_points
        self.shuffle_points = shuffle_points

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
        point_set = np.loadtxt(pts_path).astype(np.float32)
        # print(index, class_name, class_ind, pts_path)

        # # visualization
        # v = Visualizer(self.root_dir)
        # offset = self.class_to_offset_map[class_name]
        # token = meta_data["token"]
        # v.visualize(offset, token)

        if self.num_points > 0:
            if self.shuffle_points:
                choice = np.random.choice(len(point_set), self.num_points, replace=True)
            else:
                choice = np.arange(self.num_points)
            point_set = point_set[choice]

        point_set = point_set.transpose()

        # # TODO: check whether it is safe
        point_set = torch.as_tensor(point_set)
        class_ind = torch.as_tensor(class_ind)

        return point_set, class_ind

    def __len__(self):
        return len(self.meta_data)


if __name__ == "__main__":
    root_dir = "../../../data/shapenet"
    shapenet = ShapeNet(root_dir, ['train', 'val', 'test'])
    print(shapenet[0])
