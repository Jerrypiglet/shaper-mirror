import os.path as osp
import json

import numpy as np

from torch.utils.data import Dataset

from shaper.data.datasets.utils import crop_or_pad_points
from shaper_fewshot.utils.md5 import get_file_md5


class ShapeNetFewShot(Dataset):
    ROOT_DIR = "../../../data/shapenetcore_fewshot"
    train_cat_file = "train_synsetoffset2category.txt"
    target_cat_file = "target_synsetoffset2category.txt"
    split_dir = "fewshot_split"
    dataset_map = {
        "train": "shuffled_train_file_list.json",
        "val": "shuffled_val_file_list.json",
        "support_all": "shuffled_support_all_file_list.json",
        "support": "support_smp{}_cross{}_file_list.json",
        "target": "shuffled_target_file_list.json",
    }

    def __init__(self, root_dir, dataset_names, transform=None,
                 num_points=-1, shuffle_points=False,
                 k_shot=1, cross_index=0):
        self.root_dir = root_dir
        self.dataset_name = dataset_names[0]
        self.num_points = num_points
        self.shuffle_points = shuffle_points
        self.transform = transform
        self.k_shot = k_shot
        self.cross_index = cross_index

        self.md5_list = []

        if self.dataset_name in ["train", "val"]:
            self.cat_file = self.train_cat_file
        elif self.dataset_name in ["support_all", "support", "target"]:
            self.cat_file = self.target_cat_file
        else:
            raise NotImplementedError

        # classes
        self.class_to_offset_map = self._load_cat_file()
        self.offset_to_class_map = {v: k for k, v in self.class_to_offset_map.items()}
        self.classes = list(self.class_to_offset_map.keys())
        sorted(self.classes)
        self.classes_to_ind_map = {c: i for i, c in enumerate(self.classes)}

        # meta data
        self.meta_data = self._load_dataset(self.dataset_name)

        print("{} classes with {} models".format(len(self.classes), len(self.meta_data)))

    def _load_cat_file(self):
        class_to_offset_map = {}
        with open(osp.join(self.root_dir, self.cat_file), 'r') as fid:
            for line in fid:
                class_name, class_dir = line.strip().split()
                class_to_offset_map[class_name] = class_dir
        return class_to_offset_map

    def _load_pts(self, fname):
        return np.loadtxt(fname).astype(np.float32)

    def _load_dataset(self, dataset_name):
        if dataset_name == "support":
            split_fname = osp.join(self.root_dir, self.split_dir,
                                   self.dataset_map[dataset_name].format(self.k_shot, self.cross_index))
        else:
            split_fname = osp.join(self.root_dir, self.split_dir, self.dataset_map[dataset_name])
        self.md5_list.append(get_file_md5(split_fname))
        fname_list = json.load(open(split_fname, 'r'))
        meta_data = []
        for fname in fname_list:
            # fname = fname.replace("shape_data", self.root_dir)
            _, offset, token = fname.split("/")
            pts_path = osp.join(self.root_dir, offset, "points", token + '.pts')
            class_name = self.offset_to_class_map[offset]
            data = {
                'token': token,
                'class': class_name,
                'pts_path': pts_path,
            }
            meta_data.append(data)
        return meta_data

    def get_md5_list(self):
        return self.md5_list

    def __getitem__(self, index):
        meta_data = self.meta_data[index]
        class_name = meta_data["class"]
        class_ind = self.classes_to_ind_map[class_name]
        points = self._load_pts(meta_data["pts_path"])

        points, _ = crop_or_pad_points(points, self.num_points, self.shuffle_points)

        if self.transform is not None:
            points = self.transform(points)

        return {
            "points": points,
            "cls_label": class_ind,
        }

    def __len__(self):
        return len(self.meta_data)


if __name__ == "__main__":
    shapenet = ShapeNetFewShot(ShapeNetFewShot.ROOT_DIR, ["support"])
    print("The number of samples:", shapenet.__len__())
    data = shapenet[0]
    points = data["points"]
    cls_label = data["cls_label"]
    print(points.shape, points.dtype)
    print(cls_label, shapenet.classes[cls_label])

    from shaper.utils.open3d_visualize import Visualizer

    Visualizer.visualize_points(points)
