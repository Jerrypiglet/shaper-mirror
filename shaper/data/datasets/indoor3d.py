import os.path as osp

import numpy as np
import h5py
import glob
from natsort import natsorted
import torch

from torch.utils.data import Dataset


class Indoor3D(Dataset):
    room_file_list_txt = "room_filelist.txt"

    def __init__(self, root_dir, dataset_names, test_area=6, transform=None, num_points=-1, shuffle_points=False):
        self.root_dir = root_dir
        self.num_points = num_points
        self.shuffle_points = shuffle_points
        self.transform = transform

        # as dataset is small, we can load it into RAM completely
        hdf5_paths = natsorted(glob.glob(osp.join(self.root_dir, "*.h5")))
        self.point_clouds = []
        self.seg_labels = []
        for hdf5_path in hdf5_paths:
            f = h5py.File(hdf5_path)
            self.point_clouds.append(f['data'][:])
            self.seg_labels.append(f['label'][:])
        self.point_clouds = np.concatenate(self.point_clouds, 0)
        self.seg_labels = np.concatenate(self.seg_labels, 0).astype(int)

        # separate between train and test
        rooms = np.loadtxt(osp.join(self.root_dir, self.room_file_list_txt), dtype=str)
        test_ind = np.core.defchararray.startswith(rooms, "Area_{}".format(test_area))
        ind = {
            "train": np.logical_not(test_ind),
            "val": np.logical_not(test_ind),
            "test": test_ind,
        }
        keep_ind = np.zeros_like(rooms, dtype=bool)
        for dataset_name in dataset_names:
            keep_ind = np.logical_or(keep_ind, ind[dataset_name])
        self.point_clouds = self.point_clouds[keep_ind]
        self.seg_labels = self.seg_labels[keep_ind]

    def __getitem__(self, index):
        points = self.point_clouds[index]
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
        seg_labels = seg_labels[choice]

        if self.transform is not None:
            # XYZ, RGB normalized between 0 and 1, XYZ normalized w.r.t. room
            points = torch.cat(
                (self.transform(points[:, [0, 1, 2, 6, 7, 8]]), torch.Tensor(points[:, [3, 4, 5]])),
                dim=-1)
        points.transpose_(0, 1)

        out_dict = {
            "points": points,
            "seg_label": seg_labels
        }

        return out_dict

    def __len__(self):
        return len(self.point_clouds)


if __name__ == '__main__':
    dataset = Indoor3D("data/indoor3d", ("train",))
    dataset.point_clouds
