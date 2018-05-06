#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/14 12:33
# author   : QLMX
import tensorflow as tf
import numpy as np
import os

class TOOL:
    # the basic tool to convulotion
    def __init__(self, model_path=None):
        if model_path is not None:              #load parameter
            self.model = np.load(model_path, encoding='latin1').item()
        else:
            self.model = None

    def get_parameter(self, name, in_channels, out_channels, f=[3, 3], pretrain=True, conv=True):
        if self.model is not None and name in self.model.keys():
            w = self.model[name][0]
            w = np.transpose(w, (1, 0, 2, 3))
            b = self.model[name][1]
            b = b.reshape(-1)

            w_init = tf.constant_initializer(w, dtype=tf.float32)
            weights = tf.get_variable(name='weights', shape=w.shape, initializer=w_init)
            b_init = tf.constant_initializer(b, dtype=tf.float32)
            biases = tf.get_variable(name='biases', shape=b.shape, initializer=b_init)
        else:
            weights = tf.get_variable(name='weights',
                                trainable=pretrain,
                                shape=[f[0], f[1], in_channels, out_channels],
                                initializer=tf.contrib.layers.xavier_initializer())  # default is uniform distribution initialization
            if conv:
                biases = tf.get_variable(name='biases',
                                    trainable=pretrain,
                                    shape=[out_channels],
                                    initializer=tf.constant_initializer(0.0))
            else:
                biases = tf.get_variable(name='biases',
                                         trainable=pretrain,
                                         shape=[in_channels],
                                         initializer=tf.constant_initializer(0.0))
        return weights, biases

    def conv(self, name, x, out_channels, f=[3, 3], s=[1, 1, 1, 1], activation='rulu', rate=0, pretrain=True):
        '''
        the convolution layer of neural network
        :param name:the layer name
        :param x:input picture,which shape is [batch_size, hight, width, channel]
        :param out_channels:output channel
        :param f:the kernel of conv
        :param s:the strides of conv
        :param activation:the activation od convolution,which have relu, tanh, sigmoid, softmax...
        :param rate:the dialated convolution rate
        :param pretrain:Is it allowed to be trained
        :return: conv -> relu->
        '''
        in_channels = x.get_shape()[-1]
        with tf.variable_scope(name):
            w, b = self.get_parameter(name, in_channels, out_channels, f=f, pretrain=pretrain)
            if rate == 0:
                x = tf.nn.conv2d(x, w, s, padding='SAME', name='conv')
            else:
                x = tf.nn.atrous_conv2d(x, w, rate, padding='SAME', name='conv')
            x = tf.nn.bias_add(x, b, name='bias_add')
            if activation == 'relu':
                x = tf.nn.relu(x, name='relu')
            elif activation == "tanh":
                x = tf.nn.tanh(x, name='tanh')
            return x

    def deconv(self, name, x, in_channels, out_channels, output=None, f=[3,3], s=[1,2,2,1], pretrain=True):
        with tf.variable_scope(name):
            w, b = self.get_parameter(name, in_channels, out_channels, f=f, pretrain=pretrain, conv=False)
            if output is None:
                output = x.get_shape().as_list()
                output[1] *= 2
                output[2] *= 2
                output[3] = w.get_shape().as_list()[2]
            conv = tf.nn.conv2d_transpose(x, w, output, s, padding="SAME")
        return tf.nn.bias_add(conv, b)

    def pool(self, layer_name, x, kernel=[1, 2, 2, 1], s=[1, 2, 2, 1], is_max_pool=True):
        '''Pooling op
        Args:
            x: input tensor
            kernel: pooling kernel, VGG paper used [1,2,2,1], the size of kernel is 2X2
            stride: stride size, VGG paper used [1,2,2,1]
            padding:
            is_max_pool: boolen
                        if True: use max pooling
                        else: use avg pooling
        '''
        if is_max_pool:
            x = tf.nn.max_pool(x, kernel, strides=s, padding='SAME', name=layer_name)
        else:
            x = tf.nn.avg_pool(x, kernel, strides=s, padding='SAME', name=layer_name)
        return x

    def spatial_pyramid_pool(self, inputs, dimensions=[6, 3, 2, 1]):
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

    def batch_norm(self, x):
        '''Batch normlization(I didn't include the offset and scale)
        '''
        epsilon = 1e-3
        batch_mean, batch_var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x,
                                      mean=batch_mean,
                                      variance=batch_var,
                                      offset=None,
                                      scale=None,
                                      variance_epsilon=epsilon)
        return x

    def FC_layer(self, layer_name, x, out_nodes, activation = 'rulu'):
        '''Wrapper for fully connected layers with RELU activation as default
        Args:
            layer_name: e.g. 'FC1', 'FC2'
            x: input feature map
            out_nodes: number of neurons for current FC layer
        '''
        shape = x.get_shape()
        if len(shape) == 4:
            size = shape[1].value * shape[2].value * shape[3].value
        else:
            size = shape[-1].value

        with tf.variable_scope(layer_name):
            w = tf.get_variable('weights',
                                shape=[size, out_nodes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('biases',
                                shape=[out_nodes],
                                initializer=tf.constant_initializer(0.0))
            flat_x = tf.reshape(x, [-1, size])  # flatten into 1D

            x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
            if activation == 'relu':
                x = tf.nn.relu(x)
            elif activation == 'tanh':
                x = tf.nn.tanh(x)
            return x

    def conv1x1_layer(self, name, bottom, out_channels, train_mode, kernel=3, relu=True, drop=False, alpha=0.2,
                      batch=True):
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

    def convT_layer(self, name, bottom, out_channels, train_mode, relu=True, alpha=0.2, init=None):
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