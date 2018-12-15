import os.path as osp

import h5py
import numpy as np

from torch.utils.data import Dataset

from shaper_few_shot.utils.get_md5 import get_md5_for_file


class ModelNetFewShot(Dataset):
    ROOT_DIR = "/data/modelnet40/modelnet40_fewshot"
    train_cat_file = "train_shape_names.txt"
    target_cat_file = "target_shape_names.txt"
    dataset_map = {
        "train": "train_files.txt",
        "valid": "valid_files.txt",
        "support": "support_files_smp1_cross0.txt",
        "target": "target_files.txt",
    }

    def __init__(self, root_dir, dataset_names, num_per_class=1,
                 cross_num=0, transform=None,
                 shuffle_points=False, use_normal=False, num_points=-1):
        self.root_dir = root_dir
        self.dataset_name = dataset_names[0]
        self.num_points = num_points
        self.shuffle_points = shuffle_points
        self.transform = transform
        self.use_normal = use_normal

        assert num_per_class in [1, 5]
        self.num_per_class = num_per_class
        self.cross_num = cross_num
        self.MD5list = []

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
        self.data_points = []
        self.data_labels = []
        self.data_normals = []
        self._load_dataset(self.dataset_name)
        self.data_points = np.concatenate(self.data_points, axis=0)
        self.data_labels = np.concatenate(self.data_labels, axis=0)
        self.data_normals = np.concatenate(self.data_normals, axis=0)

    def _load_cat_file(self):
        with open(osp.join(self.root_dir, self.cat_file), 'r') as fid:
            classes = [line.strip() for line in fid]
        return classes

    def _load_dataset(self, dataset_name):
        if dataset_name == "support":
            file_name = "support_files_smp{}_cross{}.txt".format(self.num_per_class, self.cross_num)
            split_fname = osp.join(self.root_dir, file_name)
            # print(split_fname)
        else:
            split_fname = osp.join(self.root_dir, self.dataset_map[dataset_name])
        fname_list = [line.rstrip() for line in open(split_fname)]

        for fname in fname_list:
            data_path = osp.join(self.root_dir, osp.basename(fname))
            self.MD5list.append(get_md5_for_file(data_path))
            with h5py.File(data_path, 'r') as fid:
                num_samples = fid['label'].shape[0]
                self.data_points.append(fid['data'][:])
                self.data_normals.append(fid['normal'][:])
                self.data_labels.append(fid['label'][:].squeeze(1))
            for ind in range(num_samples):
                self.meta_data.append({
                    "offset": ind,
                    "size": num_samples,
                    "path": data_path,
                })

    def get_md5_list(self):
        return self.MD5list

    def __getitem__(self, index):
        if self.use_normal:
            points = np.concatenate((self.data_points[index], self.data_normals[index]), axis=-1)
        else:
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
    root_dir = "../../../data/modelnet40/modelnet40_fewshot"
    modelnet = ModelNetFewShot(root_dir, ['support'], cross_num=2)
    print('total data num: ', modelnet.__len__())
    # print(modelnet[0][0].size(), modelnet[0][0].type())
    # print(modelnet[0])
    # Visualizer.visualize_pts(modelnet[0][0])
