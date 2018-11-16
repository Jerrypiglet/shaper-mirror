import os.path as osp

import h5py
import numpy as np

from torch.utils.data import Dataset


class ModelNet(Dataset):
    ROOT_DIR = "/data/modelnet40"
    cat_file = "shape_names.txt"
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
        self.classes_to_ind_map = {c: i for i, c in enumerate(self.classes)}

        # load data
        self.meta_data = []
        self.data_points = []
        self.data_labels = []
        for dataset_name in dataset_names:
            self._load_dataset(dataset_name)
        self.data_points = np.concatenate(self.data_points, axis=0)
        self.data_labels = np.concatenate(self.data_labels, axis=0)

    def _load_cat_file(self):
        with open(osp.join(self.root_dir, self.cat_file), 'r') as fid:
            classes = [line.strip() for line in fid]
        return classes

    def _load_dataset(self, dataset_name):
        split_fname = osp.join(self.root_dir, self.dataset_map[dataset_name])
        fname_list = [line.rstrip() for line in open(split_fname)]

        for fname in fname_list:
            data_path = osp.join(self.root_dir, osp.basename(fname))
            with h5py.File(data_path) as fid:
                num_samples = fid['label'].shape[0]
                self.data_points.append(fid['data'][:])
                self.data_labels.append(fid['label'][:].squeeze(1))
            for ind in range(num_samples):
                self.meta_data.append({
                    "offset": ind,
                    "size": num_samples,
                    "path": data_path,
                })

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
        return len(self.meta_data)


if __name__ == "__main__":
    root_dir = "../../../data/modelnet40"
    modelnet = ModelNet(root_dir, ['test'])
    print('total data num: ', modelnet.__len__())
    # print(modelnet[0][0].size(), modelnet[0][0].type())
    # print(modelnet[0])
    # Visualizer.visualize_pts(modelnet[0][0])
