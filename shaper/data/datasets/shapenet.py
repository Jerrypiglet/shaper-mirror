import os.path as osp

import numpy as np
import h5py

from torch.utils.data import Dataset


class ShapeNet(Dataset):
    dataset_map = {
        "train": "train_hdf5_file_list.txt",
        "val": "val_hdf5_file_list.txt",
        "test": "test_hdf5_file_list.txt",
    }

    def __init__(self, root_dir, dataset_names, transform=None, num_points=-1, shuffle_points=False, load_seg=False):
        self.root_dir = root_dir
        self.num_points = num_points
        self.shuffle_points = shuffle_points
        self.transform = transform
        self.load_seg = load_seg

        # as dataset is small, we can load it into RAM completely
        hdf5_filenames = []
        for dataset_name in dataset_names:
            file_list_txt_path = osp.join(self.root_dir, self.dataset_map[dataset_name])
            with open(file_list_txt_path, 'r') as f:
                hdf5_filenames.extend(f.read().splitlines())
        self.point_clouds = []
        self.class_labels = []
        if self.load_seg:
            self.seg_labels = []
        for hdf5_filename in hdf5_filenames:
            f = h5py.File(osp.join(self.root_dir, hdf5_filename))
            self.point_clouds.append(f['data'][:])
            self.class_labels.append(f['label'][:])
            if self.load_seg:
                self.seg_labels.append(f['pid'][:])
        self.point_clouds = np.concatenate(self.point_clouds, 0)
        self.class_labels = np.concatenate(self.class_labels, 0).ravel().astype(int)
        if self.load_seg:
            self.seg_labels = np.concatenate(self.seg_labels, 0).astype(int)

    def __getitem__(self, index):
        class_label = self.class_labels[index]
        points = self.point_clouds[index]
        if self.load_seg:
            seg_labels = self.seg_labels[index]

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
        if self.load_seg:
            seg_labels = seg_labels[choice]

        if self.transform is not None:
            points = self.transform(points)

        out_dict = {
            "points": points,
            "cls_labels": class_label
        }

        if self.load_seg:
            out_dict["seg_labels"] = seg_labels

        return out_dict

    def __len__(self):
        return len(self.class_labels)
