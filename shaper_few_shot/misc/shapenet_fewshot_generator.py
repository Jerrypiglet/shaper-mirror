"""
    ShapeNetCore Few-shot dataset generator
"""
import sys
import os
import os.path as osp
import json
import random

# 4 classes for few-shot
# ["Airplane", "Car", "Motorbike", "Skateboard"]
# ["02691156", "02958343", "03790512", "04225987"]
TARGET_CLASSES = ["Airplane", "Car", "Motorbike", "Skateboard"]
SOURCE_DIR = "."


class ShapeNetFewShotGenerator(object):
    origin_cat_file = "synsetoffset2category.txt"
    origin_split_dir = "train_test_split"
    # We only use "train" and "test", "val" is skipped
    origin_dataset_map = {
        "train": "shuffled_train_file_list.json",
        # "val": "shuffled_val_file_list.json",
        "test": "shuffled_test_file_list.json",
    }
    fewshot_dataset_map = {
        "train": "shuffled_train_file_list.json",
        "val": "shuffled_val_file_list.json",
        "support_all": "shuffled_support_all_file_list.json",
        "target": "shuffled_target_file_list.json",
    }

    def __init__(self, target_classes, source_dir, target_dir,
                 num_per_class, cross_num, force=False):
        if osp.exists(target_dir) and (not force):
            print("Target dir already exists. Generation stop.")
            sys.exit()
        if not osp.exists(target_dir):
            os.mkdir(target_dir)

        self.target_classes = target_classes
        sorted(self.target_classes)
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.num_per_class = num_per_class
        self.cross_num = cross_num

        # origin dataset
        self.origin_class_to_offset_map = self._load_cat_file()
        self.origin_offset_to_class_map = {v: k for k, v in self.origin_class_to_offset_map.items()}

        # fewshot
        self.train_offset_map, self.target_offset_map = self.get_fewshot_offset_map(
            self.origin_class_to_offset_map, self.target_classes)
        self.train_offset_to_class_map = {v: k for k, v in self.train_offset_map.items()}
        self.target_offset_to_class_map = {v: k for k, v in self.target_offset_map.items()}

        self.train_classes = list(self.train_offset_map.keys())
        sorted(self.train_classes)
        # self.train_classes_to_ind_map = {c: i for i, c in enumerate(self.train_classes)}

    def generate(self):
        self.generate_train_dataset()
        self.generate_val_dataset()
        self.generate_support_dataset()
        self.generate_target_dataset()

    def _load_cat_file(self):
        class_to_offset_map = {}
        with open(osp.join(self.source_dir, self.origin_cat_file), 'r') as fid:
            for line in fid:
                class_name, class_dir = line.strip().split()
                class_to_offset_map[class_name] = class_dir
        return class_to_offset_map

    def get_fewshot_offset_map(self, origin_class_to_offset_map, target_classes):
        train_offset_map = {}
        target_offset_map = {}
        for k, v in origin_class_to_offset_map.items():
            if k not in target_classes:
                train_offset_map[k] = v
            else:
                target_offset_map[k] = v

        return train_offset_map, target_offset_map

    def generate_train_dataset(self):
        train_fname_list = []
        split_fname = osp.join(self.source_dir, self.origin_split_dir, self.origin_dataset_map["train"])
        fname_list = json.load(open(split_fname, 'r'))
        for fname in fname_list:
            _, offset, token = fname.split("/")
            class_name = self.origin_offset_to_class_map[offset]
            if class_name in self.train_classes:
                train_fname_list.append(fname)

        with open(osp.join(self.target_dir, self.fewshot_dataset_map["train"]), 'w') as f:
            json.dump(train_fname_list, f)

    def generate_val_dataset(self):
        val_fname_list = []
        split_fname = osp.join(self.source_dir, self.origin_split_dir, self.origin_dataset_map["test"])
        fname_list = json.load(open(split_fname, 'r'))
        for fname in fname_list:
            _, offset, token = fname.split("/")
            class_name = self.origin_offset_to_class_map[offset]
            if class_name in self.train_classes:
                val_fname_list.append(fname)

        with open(osp.join(self.target_dir, self.fewshot_dataset_map["val"]), 'w') as f:
            json.dump(val_fname_list, f)

    def generate_support_dataset(self):
        support_all_fname_list = []
        support_fname_list_per_class = {}
        for class_name in self.target_classes:
            support_fname_list_per_class[class_name] = []

        split_fname = osp.join(self.source_dir, self.origin_split_dir, self.origin_dataset_map["train"])
        fname_list = json.load(open(split_fname, 'r'))
        for fname in fname_list:
            _, offset, token = fname.split("/")
            class_name = self.origin_offset_to_class_map[offset]
            if class_name in self.target_classes:
                support_all_fname_list.append(fname)
                support_fname_list_per_class[class_name].append(fname)

        with open(osp.join(self.target_dir, self.fewshot_dataset_map["support_all"]), 'w') as f:
            json.dump(support_all_fname_list, f)

        for i in range(self.cross_num):
            support_fname_list = []
            for class_name in self.target_classes:
                sub_fname_list = random.sample(support_fname_list_per_class[class_name],
                                               self.num_per_class)
                support_fname_list.extend(sub_fname_list)

            with open(osp.join(self.target_dir, "support_smp{}_cross{}_file_list.json".format(
                    self.num_per_class, i
            )), 'w') as f:
                json.dump(support_fname_list, f)

    def generate_target_dataset(self):
        target_fname_list = []
        split_fname = osp.join(self.source_dir, self.origin_split_dir, self.origin_dataset_map["test"])
        fname_list = json.load(open(split_fname, 'r'))
        for fname in fname_list:
            _, offset, token = fname.split("/")
            class_name = self.origin_offset_to_class_map[offset]
            if class_name in self.target_classes:
                target_fname_list.append(fname)

        with open(osp.join(self.target_dir, self.fewshot_dataset_map["target"]), 'w') as f:
            json.dump(target_fname_list, f)


if __name__ == "__main__":
    generator = ShapeNetFewShotGenerator(
        target_classes=TARGET_CLASSES,
        source_dir=".",
        target_dir="fewshot_split",
        num_per_class=1,
        cross_num=10, force=False
    )
    generator.generate()
