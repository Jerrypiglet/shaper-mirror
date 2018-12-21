import os.path as osp

import h5py
import numpy as np

from torch.utils.data import Dataset

from shaper.data.datasets.utils import crop_or_pad_points
from shaper_fewshot.utils.md5 import get_file_md5


class ModelNetFewShot(Dataset):
    ROOT_DIR = "../../../data/modelnet40_fewshot"
    train_cat_file = "train_shape_names.txt"
    target_cat_file = "target_shape_names.txt"
    dataset_map = {
        "train": "train_files.txt",
        "valid": "valid_files.txt",
        "support": "support_files_smp1_cross0.txt",
        "target": "target_files.txt",
    }

    def __init__(self, root_dir, dataset_names, transform=None,
                 num_points=-1, shuffle_points=False, use_normal=False,
                 k_shot=1, cross_index=0,
                 ):
        self.root_dir = root_dir
        self.dataset_name = dataset_names[0]
        self.num_points = num_points
        self.shuffle_points = shuffle_points
        self.transform = transform
        self.use_normal = use_normal

        assert k_shot in [1, 5]
        self.k_shot = k_shot
        self.cross_index = cross_index
        self.md5_list = []

        if self.dataset_name in ["train", "valid"]:
            self.cat_file = self.train_cat_file
        elif self.dataset_name in ["support", "target"]:
            self.cat_file = self.target_cat_file
        else:
            raise NotImplementedError

        self.classes = self._load_cat_file()
        self.classes_to_ind_map = {c: i for i, c in enumerate(self.classes)}

        # load data
        self.meta_data = []
        self.cache_points = []
        self.cache_label = []
        self.cache_normal = []
        self._load_dataset(self.dataset_name)
        self.cache_points = np.concatenate(self.cache_points, axis=0)
        self.cache_label = np.concatenate(self.cache_label, axis=0)
        self.cache_normal = np.concatenate(self.cache_normal, axis=0)

    def _load_cat_file(self):
        with open(osp.join(self.root_dir, self.cat_file), 'r') as fid:
            classes = [line.strip() for line in fid]
        return classes

    def _load_dataset(self, dataset_name):
        if dataset_name == "support":
            file_name = "support_files_smp{}_cross{}.txt".format(self.k_shot, self.cross_index)
            split_fname = osp.join(self.root_dir, file_name)
            # print(split_fname)
        else:
            split_fname = osp.join(self.root_dir, self.dataset_map[dataset_name])
        fname_list = [line.rstrip() for line in open(split_fname)]

        for fname in fname_list:
            data_path = osp.join(self.root_dir, osp.basename(fname))
            self.md5_list.append(get_file_md5(data_path))
            with h5py.File(data_path, 'r') as fid:
                num_samples = fid['label'].shape[0]
                self.cache_points.append(fid['data'][:])
                self.cache_normal.append(fid['normal'][:])
                self.cache_label.append(fid['label'][:].squeeze(1))
            for ind in range(num_samples):
                self.meta_data.append({
                    "offset": ind,
                    "size": num_samples,
                    "path": data_path,
                })

    def get_md5_list(self):
        return self.md5_list

    def __getitem__(self, index):
        if self.use_normal:
            points = np.concatenate((self.cache_points[index], self.cache_normal[index]), axis=-1)
        else:
            points = self.cache_points[index]
        class_ind = int(self.cache_label[index])

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
    root_dir = "../../../data/modelnet40_fewshot"
    modelnet = ModelNetFewShot(root_dir, ['support'], cross_index=0)
    print("The number of samples:", modelnet.__len__())
    data = modelnet[0]
    points = data["points"]
    cls_label = data["cls_label"]
    print(points.shape, points.dtype)
    print(cls_label, modelnet.classes[cls_label])

    from shaper.utils.open3d_visualize import Visualizer

    Visualizer.visualize_points(points)
