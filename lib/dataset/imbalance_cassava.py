# To ensure fairness, we use the same code in LDAM (https://github.com/kaidic/LDAM-DRW) to produce long-tailed CIFAR datasets.
import argparse
import ast

import torch
import torchvision
import torchvision.transforms as transforms
import pickle

from lib.config.mydefault import update_config
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import random
import cv2
#import albumentations as A
from lib.data_transform.rand_augment import RandomAugment

mean, std = [0.430316, 0.496727, 0.313513], [0.238024, 0.240754, 0.228859]


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
Normal_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            # transforms.Normalize(MEAN, STD),
        ])
Eval_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((100, 100)),
            transforms.ToTensor()])

# TODOed 在rand augment后添加get_test_transform
class IMBALANCECASSAVA(Dataset):
    cls_num = 5

    # train_set = eval(cfg.DATASET.DATASET)("train", cfg)
    # valid_set = eval(cfg.DATASET.DATASET)("valid", cfg)
    def __init__(self, label_file, cfg, mode, transform_name, imb_type='exp'):

        '''with open(label_file, 'r') as f:
            # label_file的格式, (label_file image_label)
            dataAndtarget = np.array(list(map(lambda line: line.strip().split(' '), f)))
            self.data = dataAndtarget[:, 0]
            self.targets = [int(target) for target in dataAndtarget[:, 1]]'''

        if mode=='train':

            X1, Y1 = get_img(label_file[0], mode)
            X2, Y2 = get_img(label_file[1], mode)
            X3, Y3 = get_img(label_file[2], mode)

            X11, Y11 = get_img(label_file[3], mode)
            X22, Y22 = get_img(label_file[4], mode)
            X33, Y33 = get_img(label_file[5], mode)

            X111, Y111 = get_img(label_file[6], mode)
            X222, Y222 = get_img(label_file[7], mode)
            X333, Y333 = get_img(label_file[8], mode)

            if not ((Y1==Y2).all() + (Y2==Y3).all()):
                assert print('Label donot equal!')
            if not ((Y11==Y22).all() + (Y22==Y33).all()):
                assert print('Label donot equal!')
            if not ((Y111==Y222).all() + (Y222==Y333).all()):
                assert print('Label donot equal!')

            X1 = np.concatenate([X1, X2, X3], axis=1)
            index = [i for i in range(len(Y1))]
            np.random.shuffle(index)
            X1 = X1[index]
            Y1 = Y1[index]

            X11 = np.concatenate([X11, X22, X33], axis=1)
            index = [i for i in range(len(Y11))]
            np.random.shuffle(index)
            X11 = X11[index]
            Y11 = Y11[index]

            X111 = np.concatenate([X111, X222, X333], axis=1)
            index = [i for i in range(len(Y111))]
            np.random.shuffle(index)
            X111 = X111[index]
            Y111 = Y111[index]

            X = np.concatenate([X1, X11, X111], axis=0)
            Y = np.concatenate([Y1, Y11, Y111], axis=0)
        elif mode == 'valid':
            X1, Y1 = get_img(label_file[0], mode)
            X2, Y2 = get_img(label_file[1], mode)
            X3, Y3 = get_img(label_file[2], mode)
            X1 = np.concatenate([X1, X2, X3], axis=1)
            #index = [i for i in range(len(Y1))]
            #np.random.shuffle(index)
            X = X1
            Y = Y1
        elif mode == 'mytest':
            X1, Y1 = get_img_eval(label_file[0])
            X2, Y2 = get_img_eval(label_file[1])
            X3, Y3 = get_img_eval(label_file[2])
            X1 = np.concatenate([X1, X2, X3], axis=1)
            #index = [i for i in range(len(Y1))]
            #np.random.shuffle(index)
            X = X1
            Y = Y1


        self.data = X
        self.targets = torch.from_numpy(Y).long()


        train = True if mode == "train" else False
        self.cfg = cfg
        self.train = train
        self.transform_name = transform_name
        self.dual_sample = True if cfg.TRAIN.SAMPLER.DUAL_SAMPLER.ENABLE and self.train else False
        rand_number = cfg.DATASET.IMBALANCECASSAVA.RANDOM_SEED
        if self.train:
            #self.augment = RandomAugment(N=3, M=7)
            np.random.seed(rand_number)
            random.seed(rand_number)

        print("{} Mode: Contain {} images".format(mode, len(self.data)))
        if self.dual_sample or (self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train):
            self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.cls_num)
            self.class_dict = self._get_class_dict()


    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.cls_num):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def reset_epoch(self, cur_epoch):
        self.epoch = cur_epoch

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train:
            assert self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE in ["balance", "reverse"]
            if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.cls_num - 1)
            elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)

        img, target = self.data[index], self.targets[index]
        #img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            assert print('Error, can not read data!')
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        meta = dict()

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.dual_sample:
            if self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.cls_num - 1)
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "uniform":
                sample_index = random.randint(0, self.__len__() - 1)

            sample_img_row, sample_label = self.data[sample_index], self.targets[sample_index]
            #sample_img_row = cv2.imread(sample_img_path, cv2.IMREAD_COLOR)
            if sample_img_row is None:
                assert print('Error, can not read data!')
            #sample_img_row = cv2.cvtColor(sample_img_row, cv2.COLOR_BGR2RGB)
            #if self.transform_name == 'RandomAugment':
            #    sample_img_row = self.augment(data=sample_img_row)["data"]
            sample_img = sample_img_row
            meta['sample_image'] = torch.from_numpy(sample_img).float()
            meta['sample_label'] = sample_label

        #if self.transform is not None:
            #if self.transform_name == 'RandomAugment':
                #img = self.augment(data=img)["data"]
            #img = self.transform(image=img)['image']

        return torch.from_numpy(img).float(), target, meta

    def get_num_classes(self):
        return self.cls_num

    def reset_epoch(self, epoch):
        self.epoch = epoch

    def get_annotations(self):
        annos = []
        for target in self.targets:
            annos.append({'category_id': int(target)})
        return annos

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __len__(self):
        return len(self.data)


def parse_args():
    parser = argparse.ArgumentParser(description="codes for sleep scoring")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="../../configs/sleep.yaml",
        type=str,
    )
    parser.add_argument(
        "--ar",
        help="decide whether to use auto resume",
        type=ast.literal_eval,
        dest='auto_resume',
        required=False,
        default=True,
    )

    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


