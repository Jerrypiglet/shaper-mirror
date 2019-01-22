import pickle
import numpy as np
import os.path as osp

from torch.utils.data import Dataset
from shaper.data.datasets.utils import crop_or_pad_points, normalize_points
# from utils import crop_or_pad_points, normalize_points


class ScanNet(Dataset):
    """
    ScanNet dataset

    Attributes:
    
    Notes:

    """
    URL = "https://shapenet.cs.stanford.edu/media/scannet_data_pointnet2.zip"
    ROOT_DIR = "../../../data/scannet"
    dataset_map = {
        "train": "scannet_train.pickle",
        "val": "scannet_train.pickle",
        "test": "scannet_test.pickle",
    }

    def __init__(self, root_dir, dataset_names, transform=None,
                 num_points=8192, shuffle_points=False, normalize=True,
                 seg_transform=None):
        """
        Args:
            root_dir (str): the root directory of data.
            dataset_names (list of str): the names of dataset, e.g. ["train", "test"]
            transform (object): methods to transform inputs.
            num_points (int): the number of input points. -1 means using all.
            shuffle_points (bool): whether to shuffle input points.
            normalize (bool): whether to normalize points.
            seg_transform (object): methods to transform inputs and segmentation labels.

        """
        self.root_dir = root_dir
        self.dataset_names = dataset_names
        self.num_points = num_points
        self.shuffle_points = shuffle_points
        self.normalize = normalize
        self.transform = transform
        self.seg_transform = seg_transform

        self.meta_data = []
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.label_weights = {}
        for dataset_name in dataset_names:
            self._load_dataset(dataset_name)
        self._load_label_weights()

        
    def _load_dataset(self, dataset_name):
        split_fname = osp.join(self.root_dir, self.dataset_map[dataset_name])
        with open(split_fname, 'rb') as fp:
            self.scene_points_list.extend( pickle.load(fp, encoding='bytes') )
            self.semantic_labels_list.extend( pickle.load(fp, encoding='bytes') )
        for i, pts in enumerate(self.scene_points_list):
            self.meta_data.append({
                'offset': i,
                'size': pts.shape[0],
                'path': split_fname,
                'split': dataset_name,
            })

    def _load_label_weights(self):
        # train weights
        train_weights = np.zeros(21)
        for seg in self.semantic_labels_list:
            tmp,_ = np.histogram(seg, range(22))
            train_weights += tmp
        train_weights = train_weights.astype(np.float32)
        train_weights = train_weights / np.sum(train_weights)
        train_weights = 1 / np.log(1.2 + train_weights)
        # test weights
        test_weights = np.ones(21)
        # default weights
        def_weights = np.ones(21)        
        # dictionary of weights
        self.label_weights.update(
            {'train': train_weights, 'test': test_weights, 'default': def_weights}
        )
    
    def __len__(self):
        return len(self.scene_points_list)

    def __getitem__(self, index):
        point_set = self.scene_points_list[index]
        semantic_seg = self.semantic_labels_list[index].astype(np.int32)
        
        coordmax = np.max(point_set,axis=0) # max x,y,z
        coordmin = np.min(point_set,axis=0) # min x,y,z
        smpmin = np.maximum(coordmax-[1.5,1.5,3.0], coordmin) # ??
        smpmin[2] = coordmin[2] # ??
        smpsz = np.minimum(coordmax-smpmin,[1.5,1.5,3.0])
        smpsz[2] = coordmax[2]-coordmin[2]
        isvalid = False

        for i in range(10):
            curcenter = point_set[np.random.choice(len(semantic_seg),1)[0], :] # choose random pt
            # determine region around center
            curmin = curcenter-[0.75,0.75,1.5] # magic numbers??
            curmax = curcenter+[0.75,0.75,1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            # [t/f] points in the set where all 3 coords are within curmin-0.2 and curmax+0.2
            curchoice = np.sum(( point_set >= (curmin-0.2) ) * ( point_set <= (curmax+0.2) ), axis=1) == 3
            # actual points and seg labels as determined above
            cur_point_set = point_set[curchoice,:]
            cur_semantic_seg = semantic_seg[curchoice]
            # if no points found
            if len(cur_semantic_seg) == 0:
                continue
            # mask := all coords in window
            mask = np.sum((cur_point_set >= (curmin-0.01)) * (cur_point_set <= (curmax+0.01)), axis=1) == 3
            vidx = np.ceil((cur_point_set[mask,:]-curmin) / (curmax-curmin) * [31.0,31.0,62.0])
            vidx = np.unique(vidx[:,0]*31.0*62.0 + vidx[:,1]*62.0 + vidx[:,2])
            isvalid = np.sum(cur_semantic_seg>0) / len(cur_semantic_seg) >= 0.7 and len(vidx)/31.0/31.0/62.0 >= 0.02
            if isvalid:
                break

        # replace = True?
        choice = np.random.choice(len(cur_semantic_seg), self.num_points, replace=True)
        point_set = cur_point_set[choice, :]
        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
        split = self.meta_data[index]['split']
        labelweights = self.label_weights.get(split, self.label_weights['default'])
        sample_weight = labelweights[semantic_seg]
        sample_weight *= mask

        if self.normalize:
            point_set = normalize_points(point_set)
        if self.transform is not None:
            point_set = self.transform(point_set)
            if semantic_seg is not None and self.seg_transform is not None:
                # transform weights too?
                point_set, semantic_seg = self.seg_transform(point_set, semantic_seg)
            point_set.transpose_(0, 1)

        out_dict = {'points': point_set, 'seg_label': semantic_seg, 'label_weights': sample_weight}
        return out_dict


class ScanNetWholeScene():
    """
    ScanNet Whole Scene dataset

    Attributes:
    
    Notes:

    """
    URL = "https://shapenet.cs.stanford.edu/media/scannet_data_pointnet2.zip"
    ROOT_DIR = "../../../data/scannet"
    dataset_map = {
        "train": "scannet_train.pickle",
        "val": "scannet_train.pickle",
        "test": "scannet_test.pickle",
    }

    def __init__(self, root_dir, dataset_names, transform=None,
                 num_points=8192, shuffle_points=False, normalize=True,
                 seg_transform=None):
        """
        Args:
            root_dir (str): the root directory of data.
            dataset_names (list of str): the names of dataset, e.g. ["train", "test"]
            transform (object): methods to transform inputs.
            num_points (int): the number of input points. -1 means using all.
            shuffle_points (bool): whether to shuffle input points.
            normalize (bool): whether to normalize points.
            seg_transform (object): methods to transform inputs and segmentation labels.

        """
        self.root_dir = root_dir
        self.dataset_names = dataset_names
        self.num_points = num_points
        self.shuffle_points = shuffle_points
        self.normalize = normalize
        self.transform = transform
        self.seg_transform = seg_transform

        self.meta_data = []
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.label_weights = {}
        for dataset_name in dataset_names:
            self._load_dataset(dataset_name)
        self._load_label_weights()

        
    def _load_dataset(self, dataset_name):
        split_fname = osp.join(self.root_dir, self.dataset_map[dataset_name])
        with open(split_fname, 'rb') as fp:
            self.scene_points_list.extend( pickle.load(fp, encoding='bytes') )
            self.semantic_labels_list.extend( pickle.load(fp, encoding='bytes') )
        for i, pts in enumerate(self.scene_points_list):
            self.meta_data.append({
                'offset': i,
                'size': pts.shape[0],
                'path': split_fname,
                'split': dataset_name,
            })

    def _load_label_weights(self):
        # train weights
        train_weights = np.zeros(21)
        for seg in self.semantic_labels_list:
            tmp,_ = np.histogram(seg, range(22))
            train_weights += tmp
        train_weights = train_weights.astype(np.float32)
        train_weights = train_weights / np.sum(train_weights)
        train_weights = 1 / np.log(1.2 + train_weights)
        # test weights
        test_weights = np.ones(21)
        # default weights
        def_weights = np.ones(21)        
        # dictionary of weights
        self.label_weights.update(
            {'train': train_weights, 'test': test_weights, 'default': def_weights}
        )

    def __len__(self):
        return len(self.scene_points_list)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set_ini, axis=0)
        coordmin = np.min(point_set_ini, axis=0)
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0]) / 1.5).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1]) / 1.5).astype(np.int32)
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        isvalid = False
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin + [i*1.5, j*1.5, 0]
                curmax = coordmin + [(i+1)*1.5, (j+1)*1.5, coordmax[2]-coordmin[2]]
                curchoice = np.sum((point_set_ini >= (curmin-0.2)) * (point_set_ini <= (curmax+0.2)), axis=1) == 3
                cur_point_set = point_set_ini[curchoice,:]
                cur_semantic_seg = semantic_seg_ini[curchoice]
                if len(cur_semantic_seg) == 0:
                    continue

                mask = np.sum((cur_point_set >= (curmin-0.001)) * (cur_point_set <= (curmax+0.001)), axis=1) == 3
                choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
                point_set = cur_point_set[choice, :] # Nx3
                semantic_seg = cur_semantic_seg[choice] # N
                mask = mask[choice]
                if sum(mask) / float(len(mask)) < 0.01:
                    continue

                split = self.meta_data[index]['split']
                labelweights = self.label_weights.get(split, self.label_weights['default'])
                sample_weight = labelweights[semantic_seg]
                sample_weight *= mask # N
                point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
                sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN

        point_set = np.concatenate(tuple(point_sets),axis=0)
        semantic_seg = np.concatenate(tuple(semantic_segs),axis=0)
        sample_weight = np.concatenate(tuple(sample_weights),axis=0)

        if self.normalize:
            point_set = normalize_points(point_set)
        if self.transform is not None:
            point_set = self.transform(point_set)
            if semantic_seg is not None and self.seg_transform is not None:
                point_set, semantic_seg = self.seg_transform(point_set, semantic_seg)
            # point_set.transpose_(0, 1)

        out_dict = {'points': point_set, 'seg_label': semantic_seg, 'label_weights': sample_weight}
        return out_dict


class ScannetDatasetVirtualScan():
    def __init__(self, root, npoints=8192, split='train'):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_filename = os.path.join(self.root, 'scannet_%s.pickle'%(split))
        with open(self.data_filename,'rb') as fp:
            self.scene_points_list = pickle.load(fp)
            self.semantic_labels_list = pickle.load(fp)
        if split=='train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        elif split=='test':
            self.labelweights = np.ones(21)
    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        sample_weight_ini = self.labelweights[semantic_seg_ini]
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        for i in xrange(8):
            smpidx = scene_util.virtual_scan(point_set_ini,mode=i)
            if len(smpidx)<300:
                continue
            point_set = point_set_ini[smpidx,:]
            semantic_seg = semantic_seg_ini[smpidx]
            sample_weight = sample_weight_ini[smpidx]
            choice = np.random.choice(len(semantic_seg), self.npoints, replace=True)
            point_set = point_set[choice,:] # Nx3
            semantic_seg = semantic_seg[choice] # N
            sample_weight = sample_weight[choice] # N
            point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
            semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
            sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN
        point_sets = np.concatenate(tuple(point_sets),axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
        sample_weights = np.concatenate(tuple(sample_weights),axis=0)
        return point_sets, semantic_segs, sample_weights
    def __len__(self):
        return len(self.scene_points_list)


if __name__=='__main__':
    from shaper.utils.open3d_visualize import Visualizer
    root_dir = "../../../data/scannet"
    root_dir = osp.join(osp.realpath(__file__), root_dir)
    
    print("Train ScanNet")
    scannet = ScanNet(root_dir, ['train'])
    print("The number of samples:", len(scannet))
    data = scannet[0]
    points = data['points']
    seg_label = data['seg_label']
    weights = data['label_weights']
    print("Points:")
    print(points.shape, points.dtype)
    print("Labels")
    print(seg_label.shape, seg_label.dtype)
    print("Weights")
    print(weights)
    print(weights.shape, weights.dtype)

    Visualizer.visualize_points(points)

    print("\nTest ScanNet")
    scannet = ScanNet(root_dir, ['test'])
    print("The number of samples:", len(scannet))
    data = scannet[0]
    weights = data['label_weights']
    print("Weights")
    print(weights)
    print(weights.shape, weights.dtype)

    print("\nTest ScanNet Whole Scene")
    scannet = ScanNetWholeScene(root_dir, ['test'])
    print("The number of samples:", len(scannet))
    data = scannet[0]
    points = data['points']
    seg_label = data['seg_label']
    weights = data['label_weights']
    print("Points:")
    print(points.shape, points.dtype)
    print("Labels")
    print(seg_label.shape, seg_label.dtype)
    print("Weights")
    print(weights)
    print(weights.shape, weights.dtype)

    Visualizer.visualize_points(points)
