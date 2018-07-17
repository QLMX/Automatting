#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 18-7-11 下午10:32
# @Autor : QLMX
# @Email : wenruichn@gmail.com

class Config:
    #the ori data dir
    base_dir = '/home/jrs1049/Files/QLMX/graduate/'
    data_dir = base_dir + 'data/datasets/'
    portrait_img_dir = data_dir + 'portrait/image/'
    portrait_ann_dir = data_dir + 'portrait/annotation/'
    complex_img_dir = data_dir + 'complex/image/'
    complex_ann_dir = data_dir + 'complex/annotation/'
    clother_img_dir = data_dir + 'clother/image/'
    clother_ann_dir = data_dir + 'clother/annotation/'
    street_img_dir = data_dir + 'street/image/'
    street_ann_dir = data_dir + 'street/annotation/'
    test_img_dir = data_dir + 'test/image/'
    test_ann_dir = None

    name_list = [ [portrait_img_dir, portrait_ann_dir],
                  [complex_img_dir, complex_ann_dir],
                  [clother_img_dir, clother_ann_dir],
                  [street_img_dir, street_ann_dir],]

    support_models = [ 'FC-DenseNet56', 'FC-DenseNet67', 'FC-DenseNet103', 'Encoder-Decoder', 'Encoder-Decoder-Skip', 'RefineNet-Res50', \
               'RefineNet-Res101', 'RefineNet-Res152', 'FRRN-A', 'FRRN-B', 'MobileUNet', 'MobileUNet-Skip', 'PSPNet-Res50', 'PSPNet-Res101',\
               'PSPNet-Res152', 'GCN-Res50', 'GCN-Res101', 'GCN-Res152', 'DeepLabV3-Res50', 'DeepLabV3-Res101', 'DeepLabV3-Res152',\
               'DeepLabV3_plus-Res50', 'DeepLabV3_plus-Res101', 'DeepLabV3_plus-Res152', 'custom']

    #the tranisition data dir
    tfrecord_train_dir = base_dir + 'tfrecord/train'
    tfrecord_val_dir = base_dir + 'tfrecord/val'



    #the train parameter
    num_epochs = 100
    batch_size = 1
    num_val_images = 10
    lr = 0.0001
    num_keep = 1000
    num_val_images = 10
    checkpoint_step = 10
    validation_step = 1
    continue_training = False
    mode = 'train'
    # mode = 'test'


    test_bath_size = 8

    model="FC-DenseNet103"
    class_balancing = False
    threshold = 0.5

    #the data preprocess parameter
    height = 256
    width = 256
    h_flip = False
    v_flip = False
    brightness = None
    rotation = None

cfg = Config()