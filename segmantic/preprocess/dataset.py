#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/15 1:49
# author   : QLMX
import cv2, os
import numpy as np
import sys
import random

sys.path.append('../')
from config import cfg

def readData(path):
    img = []
    ann = []
    with open(path, 'r') as f:
        for line in f.readlines():
            item = line.strip().split(',')
            if len(item) == 1:
                img.append(item[0])
            else:
                img.append(item[0])
                ann.append(item[1])
    return img, ann

# Get a list of the training, validation, and testing file paths
def prepare_data(dataset_dir=cfg.data_dir):
    train_img, train_ann = readData(dataset_dir + 'train.txt')
    val_img, val_ann = readData(dataset_dir + 'val.txt')
    test_img, test_ann = readData(dataset_dir + 'test.txt')

    return train_img, train_ann, val_img, val_ann, test_img, test_ann

def load_image(path):
    image = cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB)
    return image

def data_augmentation(input_image, output_image):
    # Data augmentation
    input_image, output_image = utils.random_crop(input_image, output_image, args.crop_height, args.crop_width)

    if args.h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if args.v_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if args.brightness:
        factor = 1.0 + random.uniform(-1.0*args.brightness, args.brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if args.rotation:
        angle = random.uniform(-1*args.rotation, args.rotation)
    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)

    return input_image, output_image