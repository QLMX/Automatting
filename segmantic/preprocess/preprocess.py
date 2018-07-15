#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 18-7-11 下午10:32
# @Autor : QLMX
# @Email : wenruichn@gmail.com
import os
import sys,cv2
import random
import numpy as np

sys.path.append('../')
form config import cfg


def saveName(data_list, path):
    with open(path, 'w') as f:
        for line in data_list:
            f.write(line + '\n')

# Get a list of the training, validation, and testing file paths
def mergeData(val_rate, train_dir_list=cfg.name_dict, test_dir=cfg.test_img_dir, save_dir=cfg.data_dir):
    data = []
    test = []

    random.seed(10)
    for name in train_dir_list:
        sub_img = []
        sub_ann = []
        for file in os.listdir(name[0]):
            sub_img.append(name[0] + file)
        for file in os.listdir(name[1]):
            sub_ann.append(name[0] + file)
        sub_img.sort()
        sub_ann.sort()
        for i in range(len(sub_img)):
            img_name = sub_img[i]
            ann_name = sub_ann[i]
            data.append(img_name + ',' + ann_name)

    for file in os.listdir(test_dir):
        test.append(test_dir + file)

    val_num = int(val_rate * len(data))
    random.shuffle(data)
    random.shuffle(test)

    train = data[:val_num]
    val = data[-val_num:]

    saveName(train, save_dir + 'train.txt')
    saveName(val, save_dir + 'val.txt')
    saveName(test, save_dir + 'test.txt')


if __name__ == '__main__':
    mergeData(0.05)