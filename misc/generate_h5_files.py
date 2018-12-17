import os
import os.path as osp
import argparse

import numpy as np

import h5py


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="/media/rayc/Data/datasets/3D_datasets/ShapeNetCorev2/",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        default="/home/rayc/Projects/shaper/data/shapenet55_h5/",
        type=str,
    )
    parser.add_argument(
        "--num_pts",
        default=2048,
        type=int,
    )
    parser.add_argument(
        "--num_per_file",
        default=2048,
        type=int,
    )
    parser.add_argument(
        "--train_ratio",
        default=0.8,
        type=float,
    )

    args = parser.parse_args()
    return args


class ShapeNet55(object):
    cat_file = "/home/rayc/Projects/shaper/data/shapenet55_h5/synsetoffset2category.txt"

    def __init__(self, data_path, out_dir, num_pts, num_per_file, train_ratio):
        self.data_path = data_path
        if not osp.exists(out_dir):
            os.makedirs(out_dir)
        self.out_dir = out_dir
        self.num_pts = num_pts
        self.num_per_file = num_per_file
        self.train_ratio = train_ratio

        # classes
        self.class_to_offset_map = self._load_cat_file()
        self.offset_to_class_map = {v: k for k, v in self.class_to_offset_map.items()}

        self.classes = list(self.class_to_offset_map.keys())
        sorted(self.classes)
        self.classes_to_ind_map = {c: i for i, c in enumerate(self.classes)}
        self.offset_to_ind_map = {k: self.classes_to_ind_map[self.offset_to_class_map[k]]
                                  for _, k in self.class_to_offset_map.items()}
        self.meta_data = self._load_data()
        self.train_data, self.test_data = self._train_test_split()

    def generate(self):
        train_file_list = self._generate_h5_files(self.train_data, "data_train")
        test_file_list = self._generate_h5_files(self.test_data, "data_test")
        with open(osp.join(self.out_dir, "train_files.txt"), "w") as f:
            for file_name in train_file_list:
                f.write("{}\n".format(file_name))
        with open(osp.join(self.out_dir, "test_files.txt"), "w") as f:
            for file_name in test_file_list:
                f.write("{}\n".format(file_name))
        print("Generation finished. Files saved in {}.".format(self.out_dir))

    def _load_cat_file(self):
        class_to_offset_map = {}
        with open(self.cat_file, 'r') as fid:
            for line in fid:
                class_name, class_dir = line.strip().split()
                class_to_offset_map[class_name] = class_dir
        return class_to_offset_map

    def _load_data(self):
        data_per_class = {}
        for c in self.classes:
            data_per_class[c] = []
        xyz_file_list = []
        obj_file_list = []
        g = os.walk(self.data_path)
        for root, dirs, files in g:
            for name in files:
                # print(os.path.join(root, name))
                if name[-4:] == ".xyz":
                    xyz_file_list.append(osp.join(root, name))
                if name[-4:] == ".obj":
                    obj_file_list.append(osp.join(root, name))
        print("Pts file list: {}".format(len(xyz_file_list)))
        print("Obj file list: {}".format(len(obj_file_list)))
        assert (len(xyz_file_list) == len(obj_file_list)), "XYZ file number and OBJ file number should match."

        for xyz_file in xyz_file_list:
            # xyz_norm_id = np.loadtxt(xyz_file)
            # xyz = xyz_norm_id[:, :3]
            # norm = xyz_norm_id[:, 3:6]
            # assert (xyz_norm_id.shape[0] == self.num_pts), \
            #     "Number of points in file {} not match.".format(xyz_file)
            found = False
            for off in self.offset_to_ind_map.keys():
                if off in xyz_file.split("/"):
                    class_name = self.offset_to_class_map[off]
                    data_per_class[class_name].append(xyz_file)
                    found = True
                    break
            assert found, "Not defined class: {}".format(xyz_file)

        for k, v in data_per_class.items():
            print(k, ": ", len(v))

        return data_per_class

    def _train_test_split(self):
        train_data = {}
        test_data = {}
        for class_name, data in self.meta_data.items():
            length = len(data)
            train_length = int(length * self.train_ratio)
            train_data[class_name] = data[:train_length].copy()
            test_data[class_name] = data[train_length:].copy()
        return train_data, test_data

    def _generate_h5_files(self, data, suffix):
        file_list = []

        paths = []
        labels = []
        for class_name, class_data in data.items():
            for file_path in class_data:
                paths.append(file_path)
                labels.append([self.classes_to_ind_map[class_name]])

        data_length = len(paths)
        num_files = data_length // self.num_per_file + 1
        for file_ind in range(num_files):
            begin_idx = file_ind * self.num_per_file
            end_idx = (file_ind + 1) * self.num_per_file
            if end_idx > data_length:
                end_idx = data_length
            pts = []
            norms = []
            for i in range(begin_idx, end_idx):
                xyz_norm_id = np.loadtxt(paths[i])
                assert (xyz_norm_id.shape[0] == self.num_pts), \
                    "Number of points in file {} not match.".format(paths[i])
                pts.append([xyz_norm_id[:, :3]])
                norms.append([xyz_norm_id[:, 3:6]])
            curr_pts = np.concatenate(pts)
            curr_norms = np.concatenate(norms)
            curr_labels = np.concatenate(labels[begin_idx:end_idx])

            file_path = osp.join(self.out_dir, suffix + "_{}.h5".format(file_ind))
            with h5py.File(file_path, "w") as f:
                f.create_dataset('data', data=curr_pts)
                f.create_dataset('normal', data=curr_norms)
                f.create_dataset('label', data=curr_labels)
            file_list.append(file_path)

        return file_list


if __name__ == "__main__":
    args = parse_args()
    shapenet55 = ShapeNet55(args.path, args.out_dir, args.num_pts, args.num_per_file, args.train_ratio)
    shapenet55.generate()

    # path = args.path
    # g = os.walk(path)
    # xyz_file_list = []
    # obj_file_list = []
    # for root, dirs, files in g:
    #     for name in files:
    #         # print(os.path.join(root, name))
    #         if name[-4:] == ".xyz":
    #             xyz_file_list.append(osp.join(root, name))
    #         if name[-4:] == ".obj":
    #             obj_file_list.append(osp.join(root, name))
    # print("XYZ file list: {}".format(len(xyz_file_list)))
    # print("Obj file list: {}".format(len(obj_file_list)))
    # # assert (len(pts_file_list) == len(obj_file_list)), "XYZ file number and OBJ file number should match."
    # xyz_norm_id = np.loadtxt(xyz_file_list[0])
    # xyz = xyz_norm_id[:, :3]
    # norm = xyz_norm_id[:, 3:6]
    # id = xyz_norm_id[:, -1]
    # print("xyz: ", xyz.shape)
    # print("norm: ", norm.shape)
    # print("id: ", id.shape)

    # print("\n".join("{}".format(x) for x in pts_file_list))
