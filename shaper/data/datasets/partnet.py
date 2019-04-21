"""Partnet Segmentation

For PointNet, the authors have sampled 2048 points from ShapeNetCore to generate HDF5.
Notice that their released codes use data only verified by experts, which is same as shapenetcore_benchmark_v0.

References:
    http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html

"""

import os.path as osp
from collections import OrderedDict

import h5py
import json
import numpy as np

from torch.utils.data import Dataset
from shaper.data.datasets.utils import crop_or_pad_points, normalize_points


class PartNetH5(Dataset):
    """ Partnet Instance Segmentation

    HDF5 data has already converted catid_partid to a global seg_id.

    Attributes:


    """
    url = "https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip"
    cat_file = "all_object_categories.txt"
    seg_file = "overallid_to_catid_partid.json"
    dataset_map = {
        "train": "train_hdf5_file_list.txt",
        "val": "val_hdf5_file_list.txt",
        "test": "test_hdf5_file_list.txt",
    }

    def __init__(self, root_dir, dataset_names, transform=None,
                 num_points=-1, shuffle_points=False,
                 seg_transform=None):
        """

        Args:
            root_dir (str): the root directory of data.
            dataset_names (list of str): the names of dataset, e.g. ["train", "test"]
            transform (object): methods to transform inputs.
            num_points (int): the number of input points. -1 means using all.
            shuffle_points (bool): whether to shuffle input points.
            seg_transform (object): methods to transform inputs and segmentation labels.

        """
        self.root_dir = root_dir
        self.dataset_names = dataset_names
        self.num_points = num_points
        self.shuffle_points = shuffle_points
        self.transform = transform
        self.seg_transform = seg_transform

        self.meta_data=[]
        self.cache_points=[]
        self.cache_ins_seg_label=[]
        self.cache_point2group = []

        for dataset_name in dataset_names:
            self._load_dataset(dataset_name)

        self.cache_points = np.concatenate(self.cache_points, axis=0)
        self.cache_ins_seg_label = np.concatenate(self.cache_ins_seg_label, axis=0)
        self.cache_point2group = np.concatenate(self.cache_point2group, axis=0)


    def _load_dataset(self, dataset_name):
        split_fname = osp.join(self.root_dir, self.dataset_map[dataset_name])
        fname_list = [line.rstrip() for line in open(split_fname, 'r')]

        for fname in fname_list:
            data_path = osp.join(self.root_dir, osp.basename(fname))
            with h5py.File(data_path) as f:
                num_samples = f['label'].shape[0]
                self.cache_points.append(f['pts'][:])
                self.cache_ins_seg_label.append(f['label'][:])
                print (fname)
                print (f.keys())
                if 'point2group'+str(self.num_points) in f.keys():
                    #self.cache_point2group.append(f['point2group'+str(self.num_points)][:])
                    self.cache_point2group.append(f['point2group2500'][:])
                else:
                    self.cache_point2group.append(f['point2group10000'][:])
            json_file = data_path.replace('.h5','.json')
            with open(json_file,'r') as jf:
                self.record = json.load(jf)
            for ind in range(num_samples):
                self.meta_data.append({
                    "offset": ind,
                    "size": num_samples,
                    "path": data_path,
                })

    def __getitem__(self, index):
        points = self.cache_points[index]
        ins_seg_label = None
        ins_seg_label = self.cache_ins_seg_label[index]
        out_dict = {}

        #points, choice = crop_or_pad_points(points, self.num_points, self.shuffle_points)
        #ins_seg_label = self.cache_ins_seg_label[index][:,choice]
        ins_seg_label = ins_seg_label.astype(np.float32)


        if self.transform is not None:
            points = self.transform(points)
            if self.seg_transform is not None:
                points, ins_seg_label = self.seg_transform(points, ins_seg_label)
            points = points.transpose_(0, 1)
        else:
            points = np.transpose(points, [1,0])


        out_dict['point2group'] = self.cache_point2group[index]
        out_dict['full_points'] = points
        out_dict['full_ins_seg_label']= ins_seg_label

        out_dict["points"] = points[:,:self.num_points]
        out_dict["ins_seg_label"] = ins_seg_label[:,:self.num_points]
        out_dict['record']=self.record[index]

        return out_dict

    def __len__(self):
        return len(self.meta_data)


if __name__ == "__main__":
    # shapenet = ShapeNet("../../../data/shapenet", ["test"], load_seg=True)
    # shapenet = ShapeNetH5("../../../data/shapenet_hdf5", ["test"], load_seg=True)
    shapenet = ShapeNetNormal("../../../data/shapenet_normal", ["test"], load_seg=True)
    print("The number of samples:", shapenet.__len__())
    data = shapenet[0]
    points = data["points"]
    cls_label = data["cls_label"]
    seg_label = data["seg_label"]
    print(points.shape, points.dtype)
    print(cls_label, shapenet.classes[cls_label])
    print(seg_label.shape, seg_label.dtype)

    from shaper.utils.open3d_visualize import Visualizer

    # Visualizer.visualize_points(points)
    part_label = np.asarray([shapenet.segid_to_catid_partid_map[label][1] for label in seg_label])
    Visualizer.visualize_points_with_labels(points[:, 0:3], labels=part_label)
