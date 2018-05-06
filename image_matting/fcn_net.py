#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/14 13:06
# author   : QLMX
import tensorflow as tf
import numpy as np
from tools import TOOL


class FCN:
    '''
    the fcn_net
    '''
    def __init__(self, model_path=None, n_classes=2, pretrain=True):
        self.tools = TOOL(model_path)
        self.n_classes = n_classes
        self.pretrain = pretrain

    def build(self, x, keep_prob):
        with tf.variable_scope("FCN"):
            conv1_1 = self.tools.conv('conv1_1', x, 64, f=[3,3], s=[1,1,1,1], pretrain=self.pretrain)
            conv1_2 = self.tools.conv('conv1_2', conv1_1, 64, f=[3,3], s=[1,1,1,1], pretrain=self.pretrain)
            with tf.name_scope('pool1'):
                pool1 = self.tools.pool('pool1', conv1_2, kernel=[1,2,2,1], s=[1,2,2,1], is_max_pool=False)
            conv2_1 = self.tools.conv('conv2_1', pool1, 128, f=[3,3], s=[1,1,1,1], pretrain=self.pretrain)
            conv2_2 = self.tools.conv('conv2_2', conv2_1, 128, f=[3,3], s=[1,1,1,1], pretrain=self.pretrain)
            with tf.name_scope('pool2'):
                pool2 = self.tools.pool('pool2', conv2_2, kernel=[1,2,2,1], s=[1,2,2,1], is_max_pool=False)

            conv3_1 = self.tools.conv('conv3_1', pool2, 256, f=[3,3], s=[1,1,1,1], pretrain=self.pretrain)
            conv3_2 = self.tools.conv('conv3_2', conv3_1, 256, f=[3,3], s=[1,1,1,1], pretrain=self.pretrain)
            conv3_3 = self.tools.conv('conv3_3', conv3_2, 256, f=[3,3], s=[1,1,1,1], pretrain=self.pretrain)
            with tf.name_scope('pool3'):
                pool3 = self.tools.pool('pool3', conv3_3, kernel=[1,2,2,1], s=[1,2,2,1], is_max_pool=False)

            conv4_1 = self.tools.conv('conv4_1', pool3, 512, f=[3,3], s=[1,1,1,1], pretrain=self.pretrain)
            conv4_2 = self.tools.conv('conv4_2', conv4_1, 512, f=[3,3], s=[1,1,1,1], pretrain=self.pretrain)
            conv4_3 = self.tools.conv('conv4_3', conv4_2, 512, f=[3,3], s=[1,1,1,1], pretrain=self.pretrain)
            with tf.name_scope('pool4'):
                pool4 = self.tools.pool('pool4', conv4_3, kernel=[1,2,2,1], s=[1,2,2,1], is_max_pool=False)

            conv5_1 = self.tools.conv('conv5_1', pool4, 512, f=[3,3], s=[1,1,1,1], pretrain=self.pretrain)
            conv5_2 = self.tools.conv('conv5_2', conv5_1, 512, f=[3,3], s=[1,1,1,1], pretrain=self.pretrain)
            conv5_3 = self.tools.conv('conv5_3', conv5_2, 512, f=[3,3], s=[1,1,1,1], pretrain=self.pretrain, activation=None)
            with tf.name_scope('pool5'):
                pool5 = self.tools.pool('pool5', conv5_3, kernel=[1, 2, 2, 1], s=[1, 2, 2, 1], is_max_pool=True)

            conv6 = self.tools.conv('conv6', pool5, 4096, f=[7,7], s=[1,1,1,1], pretrain=self.pretrain)
            dropout1 = tf.nn.dropout(conv6, keep_prob)

            conv7 = self.tools.conv('conv7', dropout1, 4096, f=[1,1], s=[1,1,1,1], pretrain=self.pretrain)
            dropout2 = tf.nn.dropout(conv7, keep_prob)

            conv8 = self.tools.conv('conv8', dropout2, self.n_classes, f=[1,1], s=[1,1,1,1], activation=None, pretrain=self.pretrain)
            annotation_pred1 = tf.argmax(conv8, axis=3, name="prediction1")


            #deconvolution
            deconv_shape1 = pool4.get_shape()
            deconv1 = self.tools.deconv('deconv1', conv8, deconv_shape1[3].value, self.n_classes,
                                   output=tf.shape(pool4), f=[4,4])
            fuse_1 = tf.add(deconv1, pool4, name="fuse_1")

            deconv_shape2 = pool3.get_shape()
            deconv2 = self.tools.deconv('deconv2', fuse_1, deconv_shape2[3].value, deconv_shape1[3].value,
                                   output=tf.shape(pool3),f=[4,4])
            fuse_2 = tf.add(deconv2, pool3, name="fuse_2")

            shape = tf.shape(x)
            deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], self.n_classes])
            # print(tf.shape(pool3))
            # print(deconv_shape3)
            deconv3 = self.tools.deconv('deconv3', fuse_2, self.n_classes, deconv_shape2[3].value,
                                   output=deconv_shape3, f=[16,16], s=[1,8,8,1])

            annotation_pred = tf.argmax(deconv3, axis=3, name="prediction")


        return tf.expand_dims(annotation_pred, dim=3), deconv3

