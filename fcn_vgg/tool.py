#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/14 9:04
import tensorflow as tf

import numpy as np
from functools import reduce

def spatial_pyramid_pool(inputs, dimensions=[6, 3, 2, 1]):
    pool_list = []
    shape = inputs.get_shape().as_list()
    for d in dimensions:
        h = shape[1]
        w = shape[2]
        ph = np.ceil(h * 1.0 / d).astype(np.int32)
        pw = np.ceil(w * 1.0 / d).astype(np.int32)
        sh = np.floor(h * 1.0 / d + 1).astype(np.int32)
        sw = np.floor(w * 1.0 / d + 1).astype(np.int32)
        pool_result = tf.nn.max_pool(inputs,
                                     ksize=[1, ph, pw, 1],
                                     strides=[1, sh, sw, 1],
                                     padding='SAME')
        pool_list.append(pool_result)
    return pool_list


def conv1x1_layer(bottom, out_channels, name, train_mode, kernel=3, relu=True, drop=False, alpha=0.2, batch=True):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(bottom, out_channels, kernel, padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        if batch:
            conv = tf.layers.batch_normalization(conv, training=train_mode)

        if relu:
            relu = tf.maximum(alpha * conv, conv)
        else:
            relu = conv

        if drop:
            drop = tf.layers.dropout(relu, rate=0.5, training=train_mode)
        else:
            drop = relu

        return drop


def convT_layer(bottom, out_channels, name, train_mode, relu=True, alpha=0.2, init=None):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d_transpose(bottom, out_channels, 3, 2, padding='same',
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        batch = tf.layers.batch_normalization(conv, training=train_mode)
        if relu:
            relu = tf.maximum(alpha * batch, batch)
        else:
            relu = batch

        return relu