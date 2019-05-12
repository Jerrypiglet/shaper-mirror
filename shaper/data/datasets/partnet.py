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
        self.point2group = 'point2group2500'
        self.shuffle_points = shuffle_points
        self.transform = transform
        self.seg_transform = seg_transform

        self.meta_data=[]
        self.h5_handlers = {}
        self.record={}
        self.num_gt_masks=0
        self.active_idx=[]
        self.inactive_idx=[]

        for dataset_name in dataset_names:
            self._load_dataset(dataset_name)



    def _load_dataset(self, dataset_name):
        split_fname = osp.join(self.root_dir, self.dataset_map[dataset_name])
        fname_list = [line.rstrip() for line in open(split_fname, 'r')]

        for fname in fname_list:
            data_path = osp.join(self.root_dir, osp.basename(fname))
            with h5py.File(data_path) as f:
                print(f['label'].shape, data_path)
                self.num_gt_masks = max(self.num_gt_masks, f['label'].shape[1])
                num_samples = f['label'].shape[0]
            json_file = data_path.replace('.h5','.json')
            f = h5py.File(data_path)
            label = np.max(np.max(f['label'][:],-1),-1)
            indices = np.arange(label.shape[0]) + len(self.meta_data)
            self.active_idx.append(indices[label])
            self.inactive_idx.append(indices[np.logical_not(label)])
            for ind in range(num_samples):
                #active = np.max(f['label'][ind])
                #if active:
                #    self.active_idx.append(len(self.meta_data))
                #else:
                #    self.inactive_idx.append(len(self.meta_data))
                self.meta_data.append({
                    "offset": ind,
                    "size": num_samples,
                    "path": data_path,
                })
            with open(json_file,'r') as jf:
                self.record[data_path] = json.load(jf)
            self.h5_handlers[data_path]=None
        self.active_idx = np.concatenate(self.active_idx)
        self.inactive_idx = np.concatenate(self.inactive_idx)
        #print (len(self.meta_data), len(self.active_idx), len(self.inactive_idx))
        #print(self.active_idx)
        #print(self.inactive_idx)
        #exit(0)
        self.len_active = len(self.active_idx)
        self.len_inactive = int(0.25*self.len_active)
        len_expedite = self.len_active+self.len_inactive
        if len_expedite < len(self.meta_data) and 'train' in self.dataset_names:
            self.length = len_expedite
        else:
            self.length = len(self.meta_data)

    def shuffle_inactive(self):
        if self.length < len(self.meta_data):
            np.random.shuffle(self.inactive_idx)

    def __getitem__(self, index):
        if self.length < len(self.meta_data):
            if index < self.len_inactive:
                index = self.inactive_idx[index]
            else:
                index = self.active_idx[index - self.len_inactive]

        meta = self.meta_data[index]
        handler = self.h5_handlers[meta['path']]
        if handler is None:
            handler = h5py.File(meta['path'],'r')
            self.h5_handlers[meta['path']] = handler



        points = handler['pts'][meta['offset'],:,:]
        ins_seg_label = handler['label'][meta['offset'],:,:]
        ins_seg_label = np.concatenate([ins_seg_label, np.zeros((self.num_gt_masks - ins_seg_label.shape[0],ins_seg_label.shape[1]))], 0)
        radius = handler['radius'][meta['offset'],:]
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


        out_dict['point2group'] = handler[self.point2group][meta['offset'],:]
        out_dict['full_points'] = points
        out_dict['full_ins_seg_label']= ins_seg_label
        #out_dict['full_radius']=radius

        out_dict["points"] = points[:,:self.num_points]
        out_dict['radius'] = radius[:self.num_points]
        out_dict["ins_seg_label"] = ins_seg_label[:,:self.num_points]
        out_dict['record']=self.record[meta['path']][meta['offset']]

        return out_dict

    def __len__(self):
        return self.length


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
