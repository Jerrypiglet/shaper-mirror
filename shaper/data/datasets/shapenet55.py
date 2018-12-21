import os.path as osp

import h5py
import numpy as np
from prettytable import PrettyTable

from torch.utils.data import Dataset


class ShapeNet55(Dataset):
    ROOT_DIR = "/home/rayc/Projects/shaper/data/shapenet55_h5"
    cat_file = "synsetoffset2category.txt"
    dataset_map = {
        "train": "train_files.txt",
        "test": "test_files.txt",
    }

    def __init__(self, root_dir, dataset_names, transform=None,
                 shuffle_points=False, num_points=-1):
        self.root_dir = root_dir
        self.datasets_names = dataset_names
        self.num_points = num_points
        self.shuffle_points = shuffle_points
        self.transform = transform

        self.classes = self._load_cat_file()
        sorted(self.classes)
        self.classes_to_ind_map = {c: i for i, c in enumerate(self.classes)}

        # load data
        self.data_points = []
        self.data_normals = []
        self.data_labels = []
        for dataset_name in dataset_names:
            self._load_dataset(dataset_name)
        self.data_points = np.concatenate(self.data_points, axis=0)
        self.data_labels = np.concatenate(self.data_labels, axis=0)
        self.data_normals = np.concatenate(self.data_normals, axis=0)

    def _load_cat_file(self):
        with open(osp.join(self.root_dir, self.cat_file), 'r') as fid:
            classes = [line.strip().split()[0] for line in fid]
        return classes

    def _load_dataset(self, dataset_name):
        split_fname = osp.join(self.root_dir, self.dataset_map[dataset_name])
        fname_list = [line.rstrip() for line in open(split_fname)]

        for fname in fname_list:
            with h5py.File(fname) as fid:
                self.data_points.append(fid['data'][:])
                self.data_normals.append(fid['normal'][:])
                self.data_labels.append(fid['label'][:])

    def __getitem__(self, index):
        points = self.data_points[index]
        class_ind = int(self.data_labels[index])

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
        return self.data_labels.shape[0]

    def get_stat(self):
        title = ["Label", "Class"]
        num_per_class = {}

        for dataset_name, split_fname in self.dataset_map.items():
            title.append(dataset_name)
            num_per_class[dataset_name] = [0] * len(self.classes)
            split_fname = osp.join(self.root_dir, self.dataset_map[dataset_name])
            fname_list = [line.rstrip() for line in open(split_fname)]
            for fname in fname_list:
                with h5py.File(fname) as fid:
                    label = fid['label']
                    for l in label:
                        num_per_class[dataset_name][l] += 1

        table = PrettyTable(title)
        table.align = 'l'
        for cls_ind in range(len(self.classes)):
            row = [cls_ind, self.classes[cls_ind]]
            for dataset_name in self.dataset_map.keys():
                row.append(num_per_class[dataset_name][cls_ind])
            table.add_row(row)

        print("ShapeNet55:\n{}".format(table))


if __name__ == "__main__":
    root_dir = "/home/rayc/Projects/shaper/data/shapenet55_h5"
    shapenet55 = ShapeNet55(root_dir, ['test'])
    shapenet55.get_stat()
    # print('total data num: ', shapenet55.__len__())
    # print(modelnet[0][0].size(), modelnet[0][0].type())
    # print(modelnet[0])
    # Visualizer.visualize_pts(modelnet[0][0])