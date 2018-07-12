#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 18-7-11 下午10:32
# @Autor : QLMX
# @Email : wenruichn@gmail.com
import argparse
from utils import str2bool

#the ori data path
base_dir = '/home/jrs1049/Files/QLMX/graduate/'
portrait_img_path = base_dir + 'data/portrait/image/'
portrait_ann_path = base_dir + 'data/portrait/annotation/'
complex_img_path = base_dir + 'data/complex/image/'
complex_ann_path = base_dir + 'data/complex/annotation/'
clother_img_path = base_dir + 'data/clother/image/'
clother_ann_path = base_dir + 'data/clother/annotation/'
street_img_path = base_dir + 'data/street/image/'
street_ann_path = base_dir + 'data/street/annotation/'
class_dict = base_dir + 'data/datasets/class_dict.csv'

name_list = ['portrait', 'complex', 'clother', 'street']

#the tranisition data path
tfrecord_train_path = base_dir + 'tfrecord/train'
tfrecord_val_path = base_dir + 'tfrecord/val'



#the parameter
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
parser.add_argument('--mode', type=str, default="train", help='Select "train", "test", or "predict" mode. \
    Note that for prediction mode you have to specify an image to run the model on.')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="data/image_matting", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=704, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=576, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=10, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change.')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle.')
parser.add_argument('--model', type=str, default="FC-DenseNet56", help='The model you are using. Currently supports:\
    FC-DenseNet56, FC-DenseNet67, FC-DenseNet103, Encoder-Decoder, Encoder-Decoder-Skip, RefineNet-Res50, RefineNet-Res101, RefineNet-Res152, \
    FRRN-A, FRRN-B, MobileUNet, MobileUNet-Skip, PSPNet-Res50, PSPNet-Res101, PSPNet-Res152, GCN-Res50, GCN-Res101, GCN-Res152, DeepLabV3-Res50 \
    DeepLabV3-Res101, DeepLabV3-Res152, DeepLabV3_plus-Res50, DeepLabV3_plus-Res101, DeepLabV3_plus-Res152, custom')
args = parser.parse_args()
