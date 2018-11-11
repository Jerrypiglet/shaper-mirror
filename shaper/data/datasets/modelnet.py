import os
import os.path as osp
import sys

import h5py
import numpy as np

import torch
from torch.utils.data import Dataset


class ModelNet(Dataset):
    ROOT_DIR = "../../../data/modelnet40"
    data_dir = "modelnet40_ply_hdf5_2048"
    dataset_map = {
        "train": "train_files.txt",
        "test": "test_files.txt",
    }

    def __init__(self, root_dir, dataset_names,
                 shuffle_points=False, num_points=-1):
        self.root_dir = root_dir
        self.datasets_names = dataset_names
        self.num_points = num_points
        self.shuffle_points = shuffle_points

        # meta data
        self.meta_data = []
        for dataset_name in dataset_names:
            meta_data = self._load_dataset(dataset_name)
            self.meta_data.extend(meta_data)

    def _load_dataset(self, dataset_name):
        files_path = osp.join(self.root_dir, self.data_dir, self.dataset_map[dataset_name])
        fname_list = [line.rstrip() for line in open(files_path)]
        meta_data = []
        for fname in fname_list:
            _, offset, token = fname.split("/")
            data_path = osp.join(self.root_dir, offset, token)
            with h5py.File(data_path) as f:
                data_length = f['label'][:].shape[0]
            meta_data.append({'offset': offset,
                              'token': token,
                              'data_length': data_length})

        return meta_data

    def __getitem__(self, index):
        total_pts_ind = 0
        for file in self.meta_data:
            total_pts_ind += file['data_length']
            if index <= total_pts_ind: break
        with h5py.File(osp.join(self.root_dir, file['offset'], file['token']), 'r') as fid:
            data = fid['data'][:]
            label = fid['label'][:]
            ind = index - (total_pts_ind - file['data_length'])
            point_set = data[ind, ...]
            class_ind = label[ind]
        if self.num_points > 0:
            if self.shuffle_points:
                choice = np.random.choice(point_set.shape[0], self.num_points, replace=False)
            else:
                choice = np.arange(self.num_points)
            point_set = point_set[choice].dtype('float32')

        # point_set = torch.as_tensor(point_set).type(torch.float32)
        # class_ind = torch.as_tensor(class_ind)

        return point_set, class_ind

    def __len__(self):
        total_pts_ind = 0
        for file in self.meta_data:
            total_pts_ind += file['data_length']
        return total_pts_ind


if __name__ == "__main__":
    root_dir = ROOT_DIR
    modelnet = ModelNet(root_dir, ['test'])
    print('total data num: ', modelnet.__len__())
    print(modelnet[0][0].size(), modelnet[0][0].type())
    print(modelnet[0])
    # Visualizer.visualize_pts(modelnet[0][0])





