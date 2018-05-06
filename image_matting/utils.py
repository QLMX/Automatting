#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/14 12:31
# author   : QLMX
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import cv2

def loss(logits, labels, regularization = 0):
    '''Compute loss
    Args:
        logits: logits tensor, [batch_size, n_classes]
        labels: one-hot labels
    '''
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        if regularization != 0:
            loss = loss + regularization * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        tf.summary.scalar(scope + '/loss', loss)
        return loss


def meanIOU(labels, predictions, num_classes):
    labels = tf.argmax(labels, axis=-1)
    #labels = tf.to_int32(labels)
    
    predictions = tf.argmax(predictions, axis=-1)
    #predictions = tf.to_int32(predictions)
    

    iou, iou_op = tf.metrics.mean_iou(labels, predictions, num_classes)
    return iou, iou_op

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        temp_y_pred = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, temp_y_pred, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def accuracy(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor,
    """
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct) * 100.0
        tf.summary.scalar(scope + '/accuracy', accuracy)
    return accuracy


def num_correct_prediction(self, logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Return:
        the number of correct predictions
    """
    correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
    correct = tf.cast(correct, tf.int32)
    n_correct = tf.reduce_sum(correct)
    return n_correct


def optimize(learning_rate, loss, var_list, debug=False):
    '''optimization, use Gradient Descent as default
    '''
    with tf.name_scope('optimizer'):
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # grads = optimizer.compute_gradients(loss_val, var_list=var_list)
        # if FLAGS.debug:
        #     # print(len(var_list))
        #     for grad, var in grads:
        #         utils.add_gradient_summary(grad, var)
        # return optimizer.apply_gradients(grads)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, var_list=var_list)
        return train_op
    
def change_size(ori_img, ann_img):
    w, h, _ = ann_img.shape
    rmat = np.reshape(ann_img, (w, h))
    amat = np.zeros((w, h, 4), dtype=np.int)
    amat[:, :, 3] = rmat * 1000
    amat[:, :, 0:3] = ori_img
    return amat
    
def view_images(ori, pre, hight, width):
    pre_img = np.array(pre)
    pre_img = pre_img.reshape([hight, width])

    ori_img = ori.reshape([hight, width, 3])
    # temp = ori_img[:, :, 0]
    # ori_img[:, :, 0] = ori_img[:, :, 2]
    # ori_img[:, :, 2] = temp

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(ori)
    plt.title("ori_img")
    #
    plt.subplot(1, 2, 2)
    plt.imshow(pre_img)
    plt.title("pre")

    plt.show()

