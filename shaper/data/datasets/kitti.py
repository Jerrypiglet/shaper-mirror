import os.path as osp
import pickle
import numpy as np

from torch.utils.data import Dataset

NUM_HEADING_BIN = 12
g_type2class = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3,
                'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7}
g_class2type = {g_type2class[t]: t for t in g_type2class}
g_type_mean_size = {'Car': np.array([3.88311640418, 1.62856739989, 1.52563191462]),
                    'Van': np.array([5.06763659, 1.9007158, 2.20532825]),
                    'Truck': np.array([10.13586957, 2.58549199, 3.2520595]),
                    'Pedestrian': np.array([0.84422524, 0.66068622, 1.76255119]),
                    'Person_sitting': np.array([0.80057803, 0.5983815, 1.27450867]),
                    'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127]),
                    'Tram': np.array([16.17150617, 2.53246914, 3.53079012]),
                    'Misc': np.array([3.64300781, 1.54298177, 1.92320313])}

def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class and residual.

    Input:
        angle: rad scalar, from 0-2pi (or -pi~pi), class center at
            0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        num_class: int scalar, number of classes N
    Output:
        class_id, int, among 0,1,...,N-1
        residual_angle: float, a number such that
            class*(2pi/N) + residual_angle = angle
    '''
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_class)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - \
                     (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle


def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2 * np.pi / float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def size2class(size, type_name):
    ''' Convert 3D bounding box size to template class and residuals.
    todo (rqi): support multiple size clusters per type.

    Input:
        size: numpy array of shape (3,) for (l,w,h)
        type_name: string
    Output:
        size_class: int scalar
        size_residual: numpy array of shape (3,)
    '''
    size_class = g_type2class[type_name]
    size_residual = size - g_type_mean_size[type_name]
    return size_class, size_residual


def class2size(pred_cls, residual):
    ''' Inverse function to size2class. '''
    mean_size = g_type_mean_size[g_class2type[pred_cls]]
    return mean_size + residual



class KITTI(Dataset):
    """
    KITTI Dataset
    """
    # ROOT_DIR = "data/kitti"         # What does this mean?
    # cat_file = ""


    # cat_file = "synsetoffset2category.txt"
    # split_dir = "train_test_split"
    dataset_map = {
        "train": "frustum_carpedcyc_train.pickle",
        "val": "frustum_carpedcyc_val.pickle",
        "rgb_detection": "frustum_carpedcyc_rgb_detection.pickle",
    }

    def __init__(self, root_dir, dataset_names, transform=None,
                 num_points=-1, shuffle_points=False,
                 random_flip=False, random_shift=False, rotate_to_center=False, ):
        """

        :param root_dir: the root directory of the dataset. In our case, it is data/kitti
        :param dataset_names: a list, "train", "val", "rgb_detection"
        :param transform:
        :param num_points:
        :param shuffle_points:
        :param random_flip:
        :param random_shift:
        :param rotate_to_center:
        """
        self.root_dir = root_dir
        self.datasets_names = dataset_names
        self.num_points = num_points
        self.shuffle_points = shuffle_points
        self.transform = transform
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center

        # Load the data from pickle file
        self.id_list        = []
        self.box2d_list     = []
        self.box3d_list     = []
        self.input_list     = []
        self.label_list     = []  # it looks like a array with only [0, 1], could be a mask?
        self.type_list      = []
        self.heading_list   = []  # What is this?
        self.size_list      = []  # What is this?
        # frustum_angle is clockwise angle from positive x-axis
        self.frustum_angle_list = []  # How do you define the x axis?

        for dataset_name in dataset_names:
            self._load_dataset(dataset_name)

    def __len__(self):
        return len(self.input_list)

    def _load_dataset(self, dataset_name):
        filename = osp.join(self.root_dir, self.dataset_map[dataset_name])
        with open(filename, 'rb') as fp:
            self.id_list.extend(pickle.load(fp, encoding='latin1'))
            self.box2d_list.extend(pickle.load(fp, encoding='latin1'))
            self.box3d_list.extend(pickle.load(fp, encoding='latin1'))
            self.input_list.extend(pickle.load(fp, encoding='latin1'))
            self.label_list.extend(pickle.load(fp, encoding='latin1'))
            self.type_list.extend(pickle.load(fp, encoding='latin1'))
            self.heading_list.extend(pickle.load(fp, encoding='latin1'))
            self.size_list.extend(pickle.load(fp, encoding='latin1'))
            self.frustum_angle_list.extend(pickle.load(fp, encoding='latin1'))

    def __getitem__(self, index):
        """ Get index-th element from the picked file dataset.
        
        :param index: 
        :return: 
        """
        # ------------------------------ INPUTS ----------------------------
        rot_angle = self.get_center_view_rot_angle(index)
        class_name = self.type_list[index]
        class_ind = g_type2class[class_name]

        # Get point cloud
        if self.rotate_to_center:
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.input_list[index]

        if self.shuffle_points:
            choice = np.random.permutation(len(point_set))
        else:
            choice = np.arange(len(point_set))
        if self.num_points > 0:
            if len(point_set) >= self.num_points:
                choice = choice[:self.num_points]
            else:
                num_pad = self.num_points - len(point_set)
                pad = np.random.permutation(choice)[:num_pad]
                choice = np.concatenate([choice, pad])
        # Resample
        point_set = point_set[choice, :]

        # ------------------------------ LABELS ----------------------------
        # Ignore the segmentation for the time being.
        # seg = self.label_list[index]
        # seg = seg[choice]

        # Get center point of 3D box
        if self.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(index)
        else:
            box3d_center = self.get_box3d_center(index)

        # Heading
        if self.rotate_to_center:
            heading_angle = self.heading_list[index] - rot_angle
        else:
            heading_angle = self.heading_list[index]

        # Size
        size_class, size_residual = size2class(self.size_list[index], self.type_list[index])

        # Data Augmentation
        if self.random_flip:
            # note: rot_angle won't be correct if we have random_flip
            # so do not use it in case of random flipping.
            if np.random.random() > 0.5:  # 50% chance flipping
                point_set[:, 0] *= -1
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle
        if self.random_shift:
            dist = np.sqrt(np.sum(box3d_center[0] ** 2 + box3d_center[1] ** 2))
            shift = np.clip(np.random.randn() * dist * 0.05, dist * 0.8, dist * 1.2)
            point_set[:, 2] += shift
            box3d_center[2] += shift

        angle_class, angle_residual = angle2class(heading_angle, NUM_HEADING_BIN)

        return {
            "points": point_set,
            "cls_labels": class_ind,
            "box3d_center": box3d_center,
            "angle_class": angle_class,
            "angle_residual": angle_residual,
            "size_class": size_class,
            "size_residual": size_residual,
            "rot_angle": rot_angle
        }

    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi / 2.0 + self.frustum_angle_list[index]

    def get_box3d_center(self, index):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (self.box3d_list[index][0, :] + self.box3d_list[index][6, :]) / 2.0
        return box3d_center

    def get_center_view_box3d_center(self, index):
        ''' Frustum rotation of 3D bounding box center. '''
        box3d_center = (self.box3d_list[index][0, :] + self.box3d_list[index][6, :]) / 2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center, 0), self.get_center_view_rot_angle(index)).squeeze()

    def get_center_view_box3d(self, index):
        ''' Frustum rotation of 3D bounding box corners. '''
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view, self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set, self.get_center_view_rot_angle(index))



if __name__ == "__main__":
    root_dir = "../../../data/kitti"
    kitti_dataset = KITTI(root_dir, ['train'])
    print('total data num: ', kitti_dataset.__len__())
    print(kitti_dataset[0])
    # Visualizer.visualize_pts

    # print(modelnet[0][0].size(), modelnet[0][0].type())
    # print(modelnet[0])
    # Visualizer.visualize_pts(modelnet[0][0])











        # index = self.id_list[index]
        # box2d = self.box2d_list[index]
        # box3d = self.box3d_list[index]
        # points = self.input_list[index]
        # label = self.label_list[index]
        # type = self.type_list[index]
        # heading = self.heading_list[index]
        # size = self.size_list[index]
        # frustum_angle = self.frustum_angle_list[index]
        #
        #
        #
        #
        #
        #
        #
        #
        # meta_data = self.meta_data[index]
        # class_name = meta_data["class"]
        # class_ind = self.classes_to_ind_map[class_name]
        # points = self._load_pts(meta_data["pts_path"])
        #
        # if self.shuffle_points:
        #     choice = np.random.permutation(len(points))
        # else:
        #     choice = np.arange(len(points))
        # if self.num_points > 0:
        #     if len(points) >= self.num_points:
        #         choice = choice[:self.num_points]
        #     else:
        #         num_pad = self.num_points - len(points)
        #         pad = np.random.permutation(choice)[:num_pad]
        #         choice = np.concatenate([choice, pad])
        # points = points[choice]
        #
        # if self.transform is not None:
        #     points = self.transform(points)





        # Return everything in the Frustum Pointnet




