#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/17 21:01
import scipy.io as scio
import skimage
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
from skimage import transform, io as skio
import os
# os.chdir("D:\Graduate\project")         #windows OS
os.chdir('/media/qlmx/Files/Graduate/project')        #linux

mean_rgb = [122.675, 116.669, 104.008]
scales = [0.6, 0.8, 1.2, 1.5]
rorations = [-45, -22, 22, 45]
gammas = [.05, 0.8, 1.2, 1.5]
IMAGE_HIGHT = 800
IMAGE_WIDTH = 600

def image_resize(path, size):
    '''
    :param path: input picture image path
    :param size: set picture size,which contain width and height
    :return: none
    '''
    for name in os.listdir(path):
        img = skio.imread(os.path.join(path, name))
        img = transform.resize(img, (size['hight'], size['width']), mode = 'reflect')
        if not os.path.exists(path+'_crop'):
            os.mkdir(path+'_crop')
        skio.imsave(os.path.join(path+'_crop', name), img)

def image_crop(label, image_path, size):
    '''
    :param label: image box label
    :param image_path: want to crop iamge path
    :size: image width and hight
    :return: none
    '''
    for line in open(label, "r"):
        crop_parms = line.strip().split(" ")
        imgname = crop_parms[0]
        if os.path.exists(os.path.join(image_path, imgname)):
            img = skio.imread(os.path.join(image_path, imgname))
            img = img[int(crop_parms[1]):int(crop_parms[2])+1, int(crop_parms[3]):int(crop_parms[4])+1, :]
            img = transform.resize(img, (size['hight'], size['width']), mode = 'reflect')
            if not os.path.exists(image_path+'_crop'):
                os.mkdir(image_path+'_crop')
            skio.imsave(os.path.join(image_path+'_crop', imgname), img)

def get_mat(data_path, train=True):
    '''
    :convert images to .mat file
    :param data_path:image path
    :param train:get train data or get test data
    :return:none
    '''
    files = os.listdir(data_path+'images_data_crop')
    if not os.path.exists(data_path+'portraitFCN_data'):
        os.mkdir(data_path+'portraitFCN_data')
    if not os.path.exists(data_path+'portraitFCN+_data'):
        os.mkdir(data_path+'portraitFCN+_data')

    if train:
        reftracker = scio.loadmat(data_path+ 'images_tracker/00047.mat')['tracker']
        refpos = np.floor(np.mean(reftracker, 0))
        xxc, yyc = np.meshgrid(np.arange(1, 1801, dtype=np.int), np.arange(1, 2001, dtype=np.int))
        # normalize x and y channels
        xxc = (xxc - 600 - refpos[0]) * 1.0 / 600
        yyc = (yyc - 600 - refpos[1]) * 1.0 / 800
        maskimg = Image.open(data_path+'fcn_vgg/meanmask.png')
        maskc = np.array(maskimg, dtype=np.float)
        maskc = np.pad(maskc, (600, 600), 'minimum')

    list = []
    for file in files:
        img_data = skio.imread(os.path.join(data_path+'images_data_crop',file))
        img_data = np.array(img_data, dtype=np.double)
        if img_data.shape[2] != 3:     # make sure images are of shape(h,w,3)
            img_data = np.array([img_data for i in range(3)])
        img = img_data
        img[:,:,0] = (img_data[:,:,2] - 104.008)/255
        img[:,:,1] = (img_data[:,:,1] - 116.669)/255
        img[:,:,2] = (img_data[:,:,0] - 122.675)/255

        name = file.split(".")[0]
        list.append(int(name))
        if train:
            try:
                desttracker = scio.loadmat(data_path+'images_tracker/' + name + '.mat')['tracker']
            except:
                print("no such file!")
        # if desttracker.shape[0] == 49:
            # warp is an inverse transform, and so src and dst must be reversed here
            tform = transform.estimate_transform('affine', desttracker + 600, reftracker + 600)
            # save org mat
            warpedxx = transform.warp(xxc, tform, output_shape=xxc.shape)
            warpedyy = transform.warp(yyc, tform, output_shape=xxc.shape)
            warpedmask = transform.warp(maskc, tform, output_shape=xxc.shape)

            warpedxx = warpedxx[600:1400, 600:1200]
            warpedyy = warpedyy[600:1400, 600:1200]
            warpedmask = warpedmask[600:1400, 600:1200]

            scio.savemat('data/test/portraitFCN_data/' + name + '.mat', {'img': img})  # 保存为训练的mat文件

            imgcpy = img
            img = np.zeros((800, 600, 6))
            img[:, :, 0:3] = imgcpy
            img[:, :, 3] = warpedxx
            img[:, :, 4] = warpedyy
            img[:, :, 5] = warpedmask
            scio.savemat('data/test/portraitFCN+_data/' + name + '.mat', {'img': img})
        else:
            print(img.shape)
            scio.savemat(data_path+'portraitFCN_data/' + name + '.mat', {'img': img})  # 保存为训练的mat文件
            imgcpy = img
            img = np.zeros((800, 600, 6))
            img[:, :, 0:3] = imgcpy
            scio.savemat(data_path+'portraitFCN+_data/' + name + '.mat', {'img': img})
    list = np.array(list).reshape(1, len(list))
    if train:
        scio.savemat(data_path+'trainlist.mat', {'trainlist': list})
    else:
        scio.savemat(data_path+'testlist.mat', {'testlist': list})

if  __name__ == "__main__":
    label = "data/fcn_vgg/crop.txt"
    imgpath = "data/test/image_data"
    testpath = "data/test/images_data"
    test_mat = 'data/test/'
    size = {'hight':IMAGE_HIGHT, 'width':IMAGE_WIDTH}

    # image_crop(label, imgpath, size)
    # image_resize(testpath, size)
    get_mat(test_mat, False)
    # list = scio.loadmat("data/fcn_vgg/trainlist.mat")#['trainlist']
    # print(list)
