#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/14 12:31
# author   : QLMX
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime
from PIL import Image
from scipy import misc
from fcn_net import FCN
from psp_net import PSP
from dataset_loader import *
import tools
import os
import utils
import cv2
os.chdir('/home/jrs1049/Desktop/QLMX/graduate')         #server
# os.chdir('/media/qlmx/Files/Graduate/project')        #linux

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/psp_net/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "data/fcn_vgg/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "model/fcn_vgg", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_PATH = "/home/jrs1049/Desktop/QLMX/graduate/model/vgg19.npy"

MAX_ITERATION = int(1e4 + 1)
NUM_OF_CLASSESS = 2
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800


def train():
    #config tensorflow gup
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")

    #load fcn
    # fcn_net = FCN(model_path=MODEL_PATH, n_classes=NUM_OF_CLASSESS)
    # pred_annotation, logits = fcn_net.build(image, keep_probability)

    psp_net = PSP(model_path=MODEL_PATH, n_classes=NUM_OF_CLASSESS)
    pred_annotation, logits, preds = psp_net.build(image, keep_probability)

#     tf.summary.image("input_image", image, max_outputs=2)
#     tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
#     tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
#     loss = utils.loss(logits, tf.squeeze(annotation, squeeze_dims=[3]))
#     mean_iou, mean_iou_up = utils.meanIOU(annotation, pred_annotation, NUM_OF_CLASSESS)
#
#     tf.summary.scalar("entropy", loss)
#
#     trainable_var = tf.trainable_variables()
#     train_op = utils.optimize(FLAGS.learning_rate, loss, trainable_var)
#
#     print("Setting up summary op...")
#     summary_op = tf.summary.merge_all()
#
#     print("Setting up data reader...")
#     train_dataset_reader = BatchDatset('data/fcn_vgg/trainlist.mat')
#     validation_dataset_reader = ValDataset('data/fcn_vgg/testlist.mat')
#     test_dataset_reader = TestDataset('data/test/testlist.mat')
#
#     print("Setting up Saver...")
#     sess = tf.Session(config=config)
#     saver = tf.train.Saver()
#     summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)
#
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())
#     ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
#     if ckpt and ckpt.model_checkpoint_path:
#         saver.restore(sess, ckpt.model_checkpoint_path)
#         print("Model restored...")
#
#     #start to train
#     if FLAGS.mode == "train":
#         for itr in range(MAX_ITERATION):
#             ori_images, train_images, train_annotations = train_dataset_reader.next_batch()
#             feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.5}
#
#
#             sess.run(train_op, feed_dict=feed_dict)
#
#
#             if itr % 10 == 0:
#                 train_loss, _, iou, summary_str = sess.run([loss, mean_iou_up, mean_iou, summary_op], feed_dict=feed_dict)
#                 print("Step: %d, Train_loss:%g, Mean_Iou is %g" % (itr, train_loss, iou))
#                 summary_writer.add_summary(summary_str, itr)
#                 # train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
#                 # print("Step: %d, Train_loss:%g" % (itr, train_loss))
#                 # summary_writer.add_summary(summary_str, itr)
# #
#             if itr % 100 == 0:
#                 rpred = sess.run(pred_annotation, feed_dict=feed_dict)
#                 # rpre = np.array(rpred[0],dtype=np.int)
#                 # rpre = rpre.reshape([IMAGE_HEIGHT, IMAGE_WIDTH])
#                 utils.view_images(ori_images[0], train_annotations[0], IMAGE_HEIGHT, IMAGE_WIDTH)
#                 # plt.imshow(rpre)
#                 # plt.show()
#                 # break
# #                 print(np.sum(rpred))
# #                 print('=============')
# #                 print(np.sum(train_annotations))
# #                 print('------------>>>')
# #
# #             if itr % 500 == 0:
# #                 valid_images, valid_annotations, valid_orgs = validation_dataset_reader.next_batch()
# #                 valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
# #                                                        keep_probability: 1.0})
# #                 print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
# #                 saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
# #     elif FLAGS.mode == "test":
# #         with tf.Session() as sess:
# #             itr = 0
# #             test_images, test_annotations, test_orgs = validation_dataset_reader.next_batch()
# #             while len(test_images) > 0:
# #                 feed_dict = {image: test_images, annotation: test_annotations, keep_probability: 0.5}
# #                 preds = sess.run(pred_annotation, feed_dict=feed_dict)
# #                 org0_im = Image.fromarray(np.uint8(test_orgs[0]))
# #                 org0_im.save('result/org%d.jpg' % itr)
# #                 org1_im = Image.fromarray(np.uint8(test_orgs[1]))
# #                 org1_im.save('result/org%d.jpg' % (itr + 1))
# #                 save_alpha_img(test_orgs[0], test_annotations[0], 'res/ann%d'%itr)
# #                 save_alpha_img(test_orgs[1], test_annotations[1], 'res/ann%d'%(itr+1))
# #                 save_alpha_img(test_orgs[0], preds[0], 'result/pre%d' % itr)
# #                 save_alpha_img(test_orgs[1], preds[1], 'result/pre%d' % (itr + 1))
# #                 test_images, test_orgs = test_dataset_reader.next_batch()
# #                 itr += 2


def save_alpha_img(org, mat, name):
    w, h, _ = mat.shape
    rmat = np.reshape(mat, (w, h))
    amat = np.zeros((w, h, 4), dtype=np.int)
    amat[:, :, 3] = rmat * 1000
    amat[:, :, 0:3] = org
    misc.imsave(name + '.png', amat)

if __name__=="__main__":
    train()