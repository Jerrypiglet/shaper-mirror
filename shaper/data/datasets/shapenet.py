import os.path as osp
import json

import numpy as np

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
                 num_points=-1, shuffle_points=False):
        self.root_dir = root_dir
        self.datasets_names = dataset_names
        self.num_points = num_points
        self.shuffle_points = shuffle_points
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
        split_fname = osp.join(self.root_dir, self.split_dir, self.dataset_map[dataset_name])
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

    def __getitem__(self, index):
        meta_data = self.meta_data[index]
        class_name = meta_data["class"]
        class_ind = self.classes_to_ind_map[class_name]
        points = self._load_pts(meta_data["pts_path"])

        if self.shuffle_points:
            choice = np.random.permutation(len(points))
        else:
            choice = np.arange(len(points))
        if self.num_points > 0:
            if len(points) >= self.num_points:
                choice = choice[:self.num_points]
            else:
                num_pad = self.num_points - len(points)
                pad = np.random.permutation(choice)[:num_pad]
                choice = np.concatenate([choice, pad])
        points = points[choice]

        if self.transform is not None:
            points = self.transform(points)

        return {
            "points": points,
            "cls_labels": class_ind,
        }

    def __len__(self):
        return len(self.meta_data)


if __name__ == "__main__":
    root_dir = "../../../data/shapenetcore"
    shapenet = ShapeNet(root_dir, ['val'])
    print('total data num: ', shapenet.__len__())
