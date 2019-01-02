"""
   ModelNet40 Few-shot dataset generator
"""

import os
import os.path as osp
import sys
import numpy as np
import h5py
import random
from itertools import chain

# 10 classes for few-shot
# ["bathtub", "bed", "chair", "desk", "dresser",
#  "monitor", "night_stand", "sofa", "table", "toilet"]
# FEWSHOT_LABEL_INDS = [1, 2, 8, 12, 14, 22, 23, 30, 33, 35]

# only for resplit
FEWSHOT_LABEL_INDS = [0, 4, 5, 7, 16, 17, 21, 25, 36, 37]
SOURCE_DIR = "/home/rayc/Projects/shaper/data/modelnet40"
# Download dataset if necessary
if not osp.exists(SOURCE_DIR):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = osp.basename(www)
    os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
    os.system('mv -r %s %s' % (zipfile[:-4], SOURCE_DIR))
    os.system('rm %s' % zipfile)


def get_data_files(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def load_data_file(filename):
    return load_h5(filename)


def set_train_label(label, fewshot_label_inds):
    for i in range(label.shape[0]):
        larger_than_all = True
        for j in range(len(fewshot_label_inds)):
            if label[i] < fewshot_label_inds[j]:
                label[i] -= j
                larger_than_all = False
                break
        if larger_than_all:
            label[i] -= len(fewshot_label_inds)
    return label


def set_fewshot_label(origin_label, fewshot_label_inds):
    """

    Args:
        origin_label: int, np.array, (n,)
        fewshot_label_inds: int, list (10,), class indices of few shot categories

    Returns:
        few_shot_label: int, list(list()),
        few_shot_indices: int, list(list()), origin instance indices
    """
    few_shot_label = []
    few_shot_indices = []
    for i in range(len(fewshot_label_inds)):
        few_shot_label.append([])
        few_shot_indices.append([])

    for i in range(origin_label.shape[0]):
        for j in range(len(fewshot_label_inds)):
            if origin_label[i] == fewshot_label_inds[j]:
                few_shot_label[j].append(j)
                few_shot_indices[j].append(i)
                # label[i] = j
                break
    return few_shot_label, few_shot_indices


def set_fewshot_label_target(origin_label, fewshot_label_inds):
    fewshot_label = []
    for i in range(origin_label.shape[0]):
        for j in range(len(fewshot_label_inds)):
            if origin_label[i] == fewshot_label_inds[j]:
                fewshot_label.append(j)
                break

    return np.array(fewshot_label)


class ModelNetFewShotGenerator(object):
    def __init__(self, fewshot_label_inds, source_dir, target_dir,
                 num_per_class, cross_num, force=False):
        if osp.exists(target_dir) and (not force):
            print("Target dir already exists. Generation stop.")
            sys.exit()
        if not osp.exists(target_dir):
            os.mkdir(target_dir)

        self.fewshot_label_inds = fewshot_label_inds
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.num_per_class = num_per_class
        self.cross_num = cross_num
        self.origin_train_h5_files = get_data_files(osp.join(source_dir, "train_files.txt"))
        self.origin_test_h5_files = get_data_files(osp.join(source_dir, "test_files.txt"))

    def generate(self):
        self.generate_train_dataset()
        self.generate_valid_dataset()
        self.generate_support_dataset()
        self.generate_target_dataset()

    def generate_train_dataset(self):
        train_data = []
        train_normal = []
        train_label = []

        for fname in self.origin_train_h5_files:
            _, _, token = fname.split("/")
            with h5py.File(osp.join(SOURCE_DIR, token), 'r') as f:
                source_data = f['data'][:]
                source_normal = f['normal'][:]
                source_label = f['label'][:]
                valid_inds = []
                for i in range(source_label.shape[0]):
                    if source_label[i] not in self.fewshot_label_inds:
                        valid_inds.append(i)
                train_data.append(source_data[valid_inds, ...])
                train_normal.append(source_normal[valid_inds, ...])
                train_label.append(source_label[valid_inds, ...])
        train_data = np.array(np.concatenate(train_data))
        train_normal = np.array(np.concatenate(train_normal))
        train_label = np.array(np.concatenate(train_label))
        train_label = set_train_label(train_label, self.fewshot_label_inds)
        assert (train_data.shape[0] == train_normal.shape[0]) and (train_data.shape[0] == train_label.shape[0])

        train_h5_file_path = osp.join(self.target_dir, 'train_data.h5')
        with h5py.File(train_h5_file_path, 'w') as f:
            f.create_dataset('data', data=train_data)
            f.create_dataset('normal', data=train_normal)
            f.create_dataset('label', data=train_label)
        with open(osp.join(self.target_dir, 'train_files.txt'), 'w') as f:
            f.write(osp.join('data', train_h5_file_path))
        print('Generate train dataset finish.')
        print('Total {} models.'.format(train_data.shape[0]))

    def generate_valid_dataset(self):
        valid_data = []
        valid_normal = []
        valid_label = []

        for fname in self.origin_test_h5_files:
            _, _, token = fname.split("/")
            with h5py.File(osp.join(SOURCE_DIR, token), 'r') as f:
                source_data = f['data'][:]
                source_normal = f['normal'][:]
                source_label = f['label'][:]
                valid_inds = []
                for i in range(source_label.shape[0]):
                    if source_label[i] not in self.fewshot_label_inds:
                        valid_inds.append(i)
                valid_data.append(source_data[valid_inds, ...])
                valid_normal.append(source_normal[valid_inds, ...])
                valid_label.append(source_label[valid_inds, ...])
        valid_data = np.array(np.concatenate(valid_data))
        valid_normal = np.array(np.concatenate(valid_normal))
        valid_label = np.array(np.concatenate(valid_label))
        valid_label = set_train_label(valid_label, self.fewshot_label_inds)
        assert (valid_data.shape[0] == valid_normal.shape[0]) and (valid_data.shape[0] == valid_label.shape[0])

        valid_h5_file_path = osp.join(self.target_dir, 'valid_data.h5')
        with h5py.File(valid_h5_file_path, 'w') as f:
            f.create_dataset('data', data=valid_data)
            f.create_dataset('normal', data=valid_normal)
            f.create_dataset('label', data=valid_label)
        with open(osp.join(self.target_dir, 'valid_files.txt'), 'w') as f:
            f.write(osp.join('data', valid_h5_file_path))
        print('Generate valid dataset finish.')
        print('Total {} models.'.format(valid_data.shape[0]))

    #
    def generate_support_dataset(self):
        support_data = []
        support_normal = []
        support_label = []

        for fname in self.origin_train_h5_files:
            _, _, token = fname.split("/")
            with h5py.File(osp.join(SOURCE_DIR, token), 'r') as f:
                source_data = f['data'][:]
                source_normal = f['normal'][:]
                source_label = f['label'][:]
                valid_inds = []
                for i in range(source_label.shape[0]):
                    if source_label[i] in self.fewshot_label_inds:
                        valid_inds.append(i)
                support_data.append(source_data[valid_inds, ...])
                support_normal.append(source_normal[valid_inds, ...])
                support_label.append(source_label[valid_inds, ...])
        support_data = np.array(np.concatenate(support_data))
        support_normal = np.array(np.concatenate(support_normal))
        support_label = np.array(np.concatenate(support_label))
        support_label_all = set_fewshot_label_target(support_label, self.fewshot_label_inds)[..., np.newaxis]
        support_label, support_indices = set_fewshot_label(support_label, self.fewshot_label_inds)
        assert (support_data.shape[0] == support_normal.shape[0])
        print('Total support data: {} models'.format(support_data.shape[0]))

        support_all_h5_file_path = osp.join(self.target_dir, 'support_data_all.h5')

        with h5py.File(support_all_h5_file_path, 'w') as f:
            f.create_dataset('data', data=support_data)
            f.create_dataset('normal', data=support_normal)
            f.create_dataset('label', data=support_label_all)
        with open(osp.join(self.target_dir, 'support_all_files.txt'), 'w') as f:
            f.write(osp.join('data', support_all_h5_file_path))

        for i in range(self.cross_num):
            support_h5_file_path = osp.join(self.target_dir,
                                            'support_data_smp{}_cross{}.h5'.format(self.num_per_class, i))
            subsmp_idx = []
            subsmp_support_label = []
            for j in range(len(support_label)):
                subsmp_support_label.append([j] * self.num_per_class)
                # idx_tmp = np.random.choice(np.arange(len(support_label[j])), self.num_per_class, replace=False)
                # idx_tmp = idx_tmp.tolist()
                subsmp_idx_tmp = random.sample(support_indices[j], self.num_per_class)
                subsmp_idx.append(subsmp_idx_tmp)

            subsmp_support_label = list(chain.from_iterable(subsmp_support_label))  # list(list()) -> list()
            subsmp_idx = list(chain.from_iterable(subsmp_idx))
            subsmp_support_data = support_data[subsmp_idx, ...]
            subsmp_support_normal = support_normal[subsmp_idx, ...]
            # subsmp_support_label = support_label[support_label]
            subsmp_support_label = np.array(subsmp_support_label)[..., np.newaxis]

            with h5py.File(support_h5_file_path, 'w') as f:
                f.create_dataset('data', data=subsmp_support_data)
                f.create_dataset('normal', data=subsmp_support_normal)
                f.create_dataset('label', data=subsmp_support_label)
            with open(osp.join(self.target_dir,
                               'support_files_smp{}_cross{}.txt'.format(self.num_per_class, i)), 'w') as f:
                f.write(osp.join('data', support_h5_file_path))

        print('Generate support dataset finish.')
        print('Cross num: {}, Num per class: {}'.format(self.cross_num, self.num_per_class))

    def generate_target_dataset(self):
        target_data = []
        target_normal = []
        target_label = []

        for fname in self.origin_test_h5_files:
            _, _, token = fname.split("/")
            with h5py.File(osp.join(SOURCE_DIR, token), 'r') as f:
                source_data = f['data'][:]
                source_normal = f['normal'][:]
                source_label = f['label'][:]
                target_inds = []
                for i in range(source_label.shape[0]):
                    if source_label[i] in self.fewshot_label_inds:
                        target_inds.append(i)
                target_data.append(source_data[target_inds, ...])
                target_normal.append(source_normal[target_inds, ...])
                target_label.append(source_label[target_inds, ...])
        target_data = np.array(np.concatenate(target_data))
        target_normal = np.array(np.concatenate(target_normal))
        target_label = np.array(np.concatenate(target_label))
        target_label = set_fewshot_label_target(target_label, self.fewshot_label_inds)[..., np.newaxis]
        assert (target_data.shape[0] == target_normal.shape[0]) and (target_data.shape[0] == target_label.shape[0])

        target_h5_file_path = osp.join(self.target_dir, 'target_data.h5')
        with h5py.File(target_h5_file_path, 'w') as f:
            f.create_dataset('data', data=target_data)
            f.create_dataset('normal', data=target_normal)
            f.create_dataset('label', data=target_label)
        with open(osp.join(self.target_dir, 'target_files.txt'), 'w') as f:
            f.write(osp.join('data', target_h5_file_path))
        print('Generate target dataset finish.')
        print('Total {} models.'.format(target_data.shape[0]))


if __name__ == "__main__":
    generator = ModelNetFewShotGenerator(
        fewshot_label_inds=FEWSHOT_LABEL_INDS,
        source_dir=SOURCE_DIR,
        target_dir='/home/rayc/Projects/shaper/data/modelnet40/modelnet40_fewshot_resplit',
        num_per_class=1,
        cross_num=10, force=False
    )
    generator.generate()
