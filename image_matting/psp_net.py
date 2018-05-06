#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/14 13:06
# author   : QLMX
import tensorflow as tf
from tools import TOOL

class PSP:
    """
        A trainable version VGG19.
        """

    def __init__(self, model_path=None, n_classes=2, pretrain=True):
        self.tools = TOOL(model_path)
        self.n_classes = n_classes
        self.pretrain = pretrain

    def build(self,  image, keep_prob):
        with tf.variable_scope("FCN"):
            self.conv1_1 = self.tools.conv("conv1_1", image, 64, f=[3,3], s=[1,1,1,1], pretrain=self.pretrain)
            self.conv1_2 = self.tools.conv("conv1_2", self.conv1_1, 64, f=[3,3], s=[1,1,1,1], pretrain=self.pretrain)
            with tf.name_scope('pool1'):
                self.pool1 = self.tools.pool('pool1',  self.conv1_2, kernel=[1, 2, 2, 1], s=[1, 2, 2, 1], is_max_pool=False)

            self.conv2_1 = self.tools.conv("conv2_1", self.pool1, 128, f=[3, 3], s=[1,1,1,1], pretrain=self.pretrain)
            self.conv2_2 = self.tools.conv("conv2_2", self.conv2_1, 128, f=[3,3], s=[1,1,1,1], pretrain=self.pretrain)
            with tf.name_scope('pool2'):
                self.pool2 = self.tools.pool('pool2', self.conv2_2, kernel=[1,2,2,1], s=[1,2,2,1],is_max_pool=False)

            self.conv3_1 = self.tools.conv("conv3_1", self.pool2, 256, f=[3, 3], s=[1,1,1,1], pretrain=self.pretrain)
            self.conv3_2 = self.tools.conv("conv3_2", self.conv3_1, 256, f=[3, 3], s=[1,1,1,1], pretrain=self.pretrain)
            self.conv3_3 = self.tools.conv("conv3_3", self.conv3_2, 256, f=[3, 3], s=[1,1,1,1], pretrain=self.pretrain)
            self.conv3_4 = self.tools.conv("conv3_4", self.conv3_3, 256, f=[3, 3], s=[1,1,1,1], pretrain=self.pretrain)
            with tf.name_scope('pool3'):
                self.pool3 = self.tools.pool('pool3', self.conv3_4, kernel=[1,2,2,1], s=[1,2,2,1],is_max_pool=False)

            self.conv4_1 = self.tools.conv("conv4_1", self.pool3, 512, f=[3, 3], s=[1,1,1,1], pretrain=self.pretrain)
            self.conv4_2 = self.tools.conv("conv4_2", self.conv4_1, 512, f=[3, 3], s=[1,1,1,1], pretrain=self.pretrain)
            self.conv4_3 = self.tools.conv("conv4_3", self.conv4_2, 512, f=[3, 3], s=[1,1,1,1], pretrain=self.pretrain)
            # self.conv4_4 = self.tools.conv("conv4_4", self.conv4_3, 512, f=[3, 3], s=[1,1,1,1], pretrain=self.pretrain)
            # self.pool4 = self.tools.pool('pool4', self.conv4_4, kernel=[1,2,2,1], s=[1,2,2,1],is_max_pool=False)
            self.conv4_4 = self.tools.conv("conv4_4", self.conv4_3, 512, f=[3, 3], s=[1,1,1,1], rate=2, pretrain=self.pretrain)

            self.conv5_1 = self.tools.conv("conv5_1", self.conv4_4, 512, f=[3, 3], s=[1,1,1,1], pretrain=self.pretrain)
            self.conv5_2 = self.tools.conv("conv5_2", self.conv5_1, 512, f=[3, 3], s=[1,1,1,1], pretrain=self.pretrain)
            self.conv5_3 = self.tools.conv("conv5_3", self.conv5_2, 512, f=[3, 3], s=[1,1,1,1], pretrain=self.pretrain)
            # self.conv5_4 = self.tools.conv("conv5_4", self.conv5_3, 512, f=[3, 3], s=[1,1,1,1], pretrain=self.pretrain)
            # self.pool5 = self.tools.pool('pool5', self.conv5_4, kernel=[1,2,2,1], s=[1,2,2,1],is_max_pool=False)
            self.conv5_4 = self.tools.conv("conv5_4", self.conv5_3, 512, f=[3, 3], s=[1,1,1,1], rate=2, pretrain=self.pretrain)

            self.spp6 = self.tools.spatial_pyramid_pool(self.conv5_4)
            self.spp6_1x1_1 = self.tools.conv1x1_layer("spp6_1x1_1",self.spp6[0], 64, self.pretrain, kernel=1,
                                                    relu=True, drop=True, alpha=0.2, batch=False)
            self.spp6_1x1_2 = self.tools.conv1x1_layer("spp6_1x1_2", self.spp6[1], 64, self.pretrain, kernel=1,
                                             relu=True, drop=True, alpha=0.2, batch=True)
            self.spp6_1x1_3 = self.tools.conv1x1_layer("spp6_1x1_3", self.spp6[2], 64, self.pretrain, kernel=1,
                                             relu=True, drop=True, alpha=0.2, batch=False)
            self.spp6_1x1_4 = self.tools.conv1x1_layer("spp6_1x1_4", self.spp6[3], 64, self.pretrain, kernel=1,
                                             relu=True, drop=True, alpha=0.2, batch=True)

            self.conv5_shape = self.conv5_4.get_shape().as_list()
            self.upsample7_1 = tf.image.resize_bilinear(self.spp6_1x1_1, [self.conv5_shape[1], self.conv5_shape[2]])
            self.upsample7_2 = tf.image.resize_bilinear(self.spp6_1x1_2, [self.conv5_shape[1], self.conv5_shape[2]])
            self.upsample7_3 = tf.image.resize_bilinear(self.spp6_1x1_3, [self.conv5_shape[1], self.conv5_shape[2]])
            self.upsample7_4 = tf.image.resize_bilinear(self.spp6_1x1_4, [self.conv5_shape[1], self.conv5_shape[2]])

            self.conv1x1_8_1 = self.tools.conv1x1_layer("conv1x1_8_1", self.conv5_4, 256, self.pretrain, kernel=1,
                                                  relu=True, drop=True, alpha=0.2, batch=True)
            # self.conv1x1_8_2 = self.conv1x1_layer(self.conv4_4, 256, "conv1x1_8_2", train_mode, kernel=1,
            #                                      relu=True, drop=True, alpha=0.2, batch=True)
            self.fuse8 = tf.concat([self.upsample7_1, self.upsample7_2, self.upsample7_3,
                                    self.upsample7_4, self.conv1x1_8_1],
                                   axis=-1, name="fuse8")

            self.conv1x1_9 = self.tools.conv1x1_layer("conv1x1_9", self.fuse8, 512,  self.pretrain, kernel=3,
                                                relu=True, drop=True, alpha=0.2, batch=True)

            self.convT_10 = self.tools.convT_layer("convT_10", self.conv1x1_9, 128, self.pretrain, relu=True, alpha=0.2)

            self.convT_11 = self.tools.convT_layer("convT_11", self.convT_10, 128, self.pretrain, relu=True, alpha=0.2)

            self.convT_12 = self.tools.convT_layer("convT_12", self.convT_11, 64, self.pretrain, relu=True, alpha=0.2)
            # self.conv1x1_12 = self.conv1x1_layer(self.convT_11, 64, "conv1x1_12", train_mode, relu=True, drop=False)

            self.convT_13 = self.tools.conv1x1_layer("convT_13", self.convT_12, 64, self.pretrain, kernel=3,
                                               relu=True, drop=False, alpha=0.2, batch=False)

            self.logits = self.tools.conv1x1_layer("logits", self.convT_13, self.n_classes, self.pretrain, kernel=3,
                                             relu=False, drop=False, batch=False)

            self.preds = tf.nn.softmax(self.logits, name='preds')

            annotation_pred = tf.argmax(self.logits, axis=3, name="prediction")
            #print("preds is {}, shape is {}".format(annotation_pred, annotation_pred.shape))
            annotation_pred = tf.expand_dims(annotation_pred, dim=3)

            # print(self.pool1.shape)
            # print(self.pool2.shape)
            # print(self.pool3.shape)
            # print(self.conv4_4.shape)
            # print(self.conv5_4.shape)
            # print(self.fuse8.shape)
            # print(self.conv1x1_9.shape)
            #
            # print(self.convT_10.shape)
            # print(self.convT_11.shape)
            # print(self.convT_12.shape)
            # print(self.convT_13.shape)
            #
            # print("logits is {}, shape is {}".format(self.logits, self.logits.shape))
            # print("preds is {}, shape is {}".format(self.preds, self.preds.shape))
            # print("anno+argmax is {}, shape is {}".format(annotation_pred, annotation_pred.shape))

        return annotation_pred, self.logits, self.preds