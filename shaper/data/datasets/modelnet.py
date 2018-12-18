import os.path as osp

import h5py
import numpy as np

from torch.utils.data import Dataset
from shaper.data.datasets.utils import crop_or_pad_points


class ModelNetH5(Dataset):
    """ModelNet HDF5 dataset

    Attributes:
        root_dir (str): the root directory of data.
        dataset_names (list of str): the names of dataset, e.g. ["train", "test"]
        transform (object): methods to transform inputs.
        num_points (int): the number of input points. -1 means using all.
        shuffle_points (bool): whether to shuffle input points.
        classes (list): the names of classes
        class_to_ind_map (dict): mapping from class names to class indices
        meta_data (list of dict): meta information of data

    """
    URL = " https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
    ROOT_DIR = "/data/modelnet40"
    cat_file = "shape_names.txt"
    dataset_map = {
        "train": "train_files.txt",
        "test": "test_files.txt",
    }

    def __init__(self, root_dir, dataset_names, transform=None,
                 num_points=-1, shuffle_points=False):
        self.root_dir = root_dir
        self.dataset_names = dataset_names
        self.num_points = num_points
        self.shuffle_points = shuffle_points
        self.transform = transform

        self.classes = self._load_cat_file()
        self.class_to_ind_map = {c: i for i, c in enumerate(self.classes)}

        # load meta data and cache
        self.meta_data = []
        self.cache_points = []
        self.cache_normal = []
        self.cache_label = []
        for dataset_name in dataset_names:
            self._load_dataset(dataset_name)

        self.cache_points = np.concatenate(self.cache_points, axis=0)
        self.cache_label = np.concatenate(self.cache_label, axis=0)
        self.cache_normal = np.concatenate(self.cache_normal, axis=0)

        print("{} classes with {} models".format(len(self.classes), len(self.meta_data)))

    def _load_cat_file(self):
        with open(osp.join(self.root_dir, self.cat_file), 'r') as fid:
            classes = [line.strip() for line in fid]
        return classes

    def _load_dataset(self, dataset_name):
        split_fname = osp.join(self.root_dir, self.dataset_map[dataset_name])
        fname_list = [line.rstrip() for line in open(split_fname)]

        for fname in fname_list:
            data_path = osp.join(self.root_dir, osp.basename(fname))
            with h5py.File(data_path) as f:
                num_samples = f['label'].shape[0]
                self.cache_points.append(f['data'][:])
                self.cache_normal.append(f['normal'][:])
                self.cache_label.append(f['label'][:].squeeze(1))
            for ind in range(num_samples):
                self.meta_data.append({
                    "offset": ind,
                    "size": num_samples,
                    "path": data_path,
                })

    def __getitem__(self, index):
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
    root_dir = "../../../data/modelnet40"
    modelnet = ModelNetH5(root_dir, ["test"])
    print("The number of samples:", modelnet.__len__())
    data = modelnet[0]
    points = data["points"]
    cls_label = data["cls_label"]
    print(points.shape, points.dtype)
    print(cls_label, modelnet.classes[cls_label])

    from shaper.utils.open3d_visualize import Visualizer

    Visualizer.visualize_points(points)
