import os.path as osp

import random
import h5py
import numpy as np

from torch.utils.data import Dataset

from shaper_few_shot.utils.get_md5 import get_md5_for_file


class ModelNetCompare(Dataset):
    ROOT_DIR = "/data/modelnet40/modelnet40_fewshot"
    train_cat_file = "train_shape_names.txt"
    target_cat_file = "target_shape_names.txt"
    dataset_map = {
        "train": "train_files.txt",
        "valid": "valid_files.txt",
        "support": "support_files_smp1_cross0.txt",
        "target": "target_files.txt",
    }
    train_class_num = 30
    target_class_num = 10

    def __init__(self, root_dir, dataset_names, class_num_per_batch=10,
                 batch_support_num_per_class=1,
                 batch_target_num=8,
                 num_per_class=1,
                 cross_num=0, transform=None,
                 shuffle_data=False,
                 shuffle_points=False, use_normal=False, num_points=-1):
        """

        :param root_dir:
        :param dataset_names:
        :param class_num_per_batch: number of classes per batch
        :param batch_support_num_per_class: number of support instance per class
        :param batch_target_num: number of target instance per batch
        :param num_per_class:
        :param cross_num:
        :param transform:
        :param shuffle_points:
        :param num_points:
        """
        self.root_dir = root_dir
        self.dataset_name = dataset_names[0]
        self.class_num_per_batch = class_num_per_batch
        self.batch_support_num_per_class = batch_support_num_per_class
        self.batch_target_num = batch_target_num
        self.num_points = num_points
        self.shuffle_data = shuffle_data
        self.shuffle_points = shuffle_points
        self.transform = transform
        self.use_normal = use_normal

        assert num_per_class in [1, 5]
        self.num_per_class = num_per_class
        self.cross_num = cross_num

        if self.dataset_name in ["train", "valid"]:
            self.cat_file = self.train_cat_file
        elif self.dataset_name in ["support", "target"]:
            self.cat_file = self.target_cat_file
        else:
            raise NotImplementedError

        self.classes = self._load_cat_file()
        self.classes_to_ind_map = {c: i for i, c in enumerate(self.classes)}

        # load data
        self.total_support_num = 0
        self.total_target_num = 0
        self.support_MD5list = []
        self.target_MD5list = []
        self.data_points = []
        self.data_labels = []
        self.data_normals = []
        self.total_target_data_points = []
        self.total_target_data_labels = []
        self.total_target_data_normals = []
        self._load_dataset(self.dataset_name)
        # self.data_points = np.concatenate(self.data_points, axis=0)
        # self.data_labels = np.concatenate(self.data_labels, axis=0)
        # self.data_normals = np.concatenate(self.data_normals, axis=0)

    def _load_cat_file(self):
        with open(osp.join(self.root_dir, self.cat_file), 'r') as fid:
            classes = [line.strip() for line in fid]
        return classes

    def _load_dataset(self, dataset_name):
        if dataset_name == "support":
            file_name = "support_files_smp{}_cross{}.txt".format(self.num_per_class, self.cross_num)
            support_split_fname = osp.join(self.root_dir, file_name)
            target_split_fname = support_split_fname
            class_num = self.target_class_num
            # print(split_fname)
        elif dataset_name == "target":
            file_name = "support_files_smp{}_cross{}.txt".format(self.num_per_class, self.cross_num)
            support_split_fname = osp.join(self.root_dir, file_name)
            target_split_fname = osp.join(self.root_dir, self.dataset_map[dataset_name])
            class_num = self.target_class_num
        elif dataset_name == "train":
            support_split_fname = osp.join(self.root_dir, self.dataset_map[dataset_name])
            target_split_fname = support_split_fname
            class_num = self.train_class_num
        else:
            support_split_fname = osp.join(self.root_dir, self.dataset_map["train"])
            target_split_fname = osp.join(self.root_dir, self.dataset_map["valid"])
            class_num = self.train_class_num

        support_fname_list = [line.rstrip() for line in open(support_split_fname)]
        support_data_points = []
        support_data_normals = []
        support_data_labels = []

        for fname in support_fname_list:
            data_path = osp.join(self.root_dir, osp.basename(fname))
            self.support_MD5list.append(get_md5_for_file(data_path))
            with h5py.File(data_path, 'r') as fid:
                num_samples = fid['label'].shape[0]
                support_data_points.append(fid['data'][:])
                support_data_normals.append(fid['normal'][:])
                support_data_labels.append(fid['label'][:].squeeze(1))
        support_data_points = np.concatenate(support_data_points, axis=0)
        support_data_normals = np.concatenate(support_data_normals, axis=0)
        support_data_labels = np.concatenate(support_data_labels, axis=0)
        self.total_support_num = support_data_labels.shape[0]

        support_data_points_per_class = {}
        support_data_normals_per_class = {}
        for i in range(class_num):
            support_data_points_per_class[i] = []
            support_data_normals_per_class[i] = []
        for i in range(support_data_labels.shape[0]):
            support_data_points_per_class[support_data_labels[i]].append(support_data_points[i, ...])
            support_data_normals_per_class[support_data_labels[i]].append(support_data_normals[i, ...])

        if target_split_fname == support_split_fname:
            target_data_points_per_class = support_data_points_per_class
            target_data_normals_per_class = support_data_normals_per_class
        else:
            target_data_points_per_class = {}
            target_data_normals_per_class = {}
            target_fname_list = [line.rstrip() for line in open(target_split_fname)]
            target_data_points = []
            target_data_normals = []
            target_data_labels = []

            for fname in target_fname_list:
                data_path = osp.join(self.root_dir, osp.basename(fname))
                self.target_MD5list.append(get_md5_for_file(data_path))
                with h5py.File(data_path, 'r') as fid:
                    num_samples = fid['label'].shape[0]
                    target_data_points.append(fid['data'][:])
                    target_data_normals.append(fid['normal'][:])
                    target_data_labels.append(fid['label'][:].squeeze(1))
            target_data_points = np.concatenate(target_data_points, axis=0)
            target_data_normals = np.concatenate(target_data_normals, axis=0)
            target_data_labels = np.concatenate(target_data_labels, axis=0)
            self.total_target_num = target_data_labels.shape[0]
            for i in range(class_num):
                target_data_points_per_class[i] = []
                target_data_normals_per_class[i] = []
            for i in range(target_data_labels.shape[0]):
                target_data_points_per_class[target_data_labels[i]].append(target_data_points[i, ...])
                target_data_normals_per_class[target_data_labels[i]].append(target_data_normals[i, ...])

        # support_min_length = len(support_data_points_per_class[0])
        # for i in range(len(support_data_points_per_class)):
        #     random.shuffle(support_data_points_per_class[i]) # shuffle data order
        #     if len(support_data_points_per_class[i]) < support_min_length:
        #         support_min_length = len(support_data_points_per_class[i])
        #
        # target_min_length = len(target_data_points_per_class[0])
        # for i in range(len(target_data_points_per_class)):
        #     random.shuffle(target_data_points_per_class[i])  # shuffle data order
        #     if len(target_data_points_per_class[i]) < target_min_length:
        #         target_min_length = len(target_data_points_per_class[i])

        self.class_batch_num = class_num // self.class_num_per_batch
        self.batch_num = 0  # total batch number, depends on the target instance number

        shuffle_class_inds = np.arange(class_num)
        if self.shuffle_data:
            np.random.shuffle(shuffle_class_inds)

        for class_batch in range(self.class_batch_num):
            class_inds = shuffle_class_inds[
                         class_batch * self.class_num_per_batch:(class_batch + 1) * self.class_num_per_batch]

            # construct target data set
            target_data_labels_set = []
            target_data_points_set = []
            target_data_normals_set = []
            for class_ind in class_inds:
                target_data_points_set.extend(target_data_points_per_class[class_ind])
                target_data_normals_set.extend(target_data_normals_per_class[class_ind])
                target_data_labels_set.extend([class_ind] * len(target_data_normals_per_class[class_ind]))

            target_data_length = len(target_data_points_set)
            shuffle_inds = np.arange(target_data_length)
            if self.shuffle_data:
                np.random.shuffle(shuffle_inds)
                shuffled_target_data_labels_set = []
                shuffled_target_data_points_set = []
                shuffled_target_data_normals_set = []
                for i in range(target_data_length):
                    shuffled_target_data_labels_set.append(target_data_labels_set[shuffle_inds[i]])
                    shuffled_target_data_points_set.append(target_data_points_set[shuffle_inds[i]])
                    shuffled_target_data_normals_set.append(target_data_normals_set[shuffle_inds[i]])
            else:
                shuffled_target_data_labels_set = target_data_labels_set.copy()
                shuffled_target_data_points_set = target_data_points_set.copy()
                shuffled_target_data_normals_set = target_data_normals_set.copy()

            curr_batch_num = target_data_length // self.batch_target_num
            self.batch_num += curr_batch_num

            for i in range(curr_batch_num):
                # Add support data
                for j in range(len(class_inds)):
                    # random_inds = random.randint(0, len(support_data_points_per_class[class_inds[j]]) - 1)
                    assert (len(support_data_points_per_class[class_inds[j]]) >= self.batch_support_num_per_class
                            ), "The data number of class [{}] is smaller than the batch_support_num_per_class [{}]"\
                        .format(len(support_data_points_per_class[class_inds[j]]), self.batch_support_num_per_class)
                    random_inds = np.random.choice(len(support_data_points_per_class[class_inds[j]]),
                                                   size=self.batch_support_num_per_class, replace=False)
                    for random_ind in random_inds:
                        self.data_labels.append(class_inds[j])
                        self.data_points.append(support_data_points_per_class[class_inds[j]][random_ind])
                        self.data_normals.append(support_data_normals_per_class[class_inds[j]][random_ind])
                # Add target data
                for j in range(self.batch_target_num):
                    curr_inds = i * self.batch_target_num + j
                    self.data_labels.append(shuffled_target_data_labels_set[curr_inds])
                    self.data_points.append(shuffled_target_data_points_set[curr_inds])
                    self.data_normals.append(shuffled_target_data_normals_set[curr_inds])
                    self.total_target_data_labels.append(shuffled_target_data_labels_set[curr_inds])
                    self.total_target_data_points.append(shuffled_target_data_points_set[curr_inds])
                    self.total_target_data_normals.append(shuffled_target_data_normals_set[curr_inds])

    def get_md5_list(self):
        return {"support": self.support_MD5list,
                "target": self.target_MD5list}

    def __getitem__(self, index):
        if not self.use_normal:
            points = self.data_points[index]
        else:
            points = np.concatenate((self.data_points[index], self.data_normals[index]), axis=-1)
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

    def reset(self):
        self.total_support_num = 0
        self.total_target_num = 0
        self.support_MD5list = []
        self.target_MD5list = []
        self.data_points = []
        self.data_labels = []
        self.data_normals = []
        self.total_target_data_points = []
        self.total_target_data_labels = []
        self.total_target_data_normals = []
        self._load_dataset(self.dataset_name)

    def __len__(self):
        return self.batch_num * (self.class_num_per_batch * self.batch_support_num_per_class + self.batch_target_num)

    def get_batch_size(self):
        return self.class_num_per_batch * self.batch_support_num_per_class + self.batch_target_num

    def get_total_target_num(self):
        return self.total_target_num


if __name__ == "__main__":
    root_dir = "../../../data/modelnet40/modelnet40_fewshot"
    modelnet = ModelNetCompare(root_dir, ['train'], cross_num=2, batch_target_num=10)
    print('total data num: ', modelnet.__len__())
    # print(modelnet[0][0].size(), modelnet[0][0].type())
    # print(modelnet[0])
    # Visualizer.visualize_pts(modelnet[0][0])
