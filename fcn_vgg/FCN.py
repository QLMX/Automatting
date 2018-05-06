#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/19 21:31
from __future__ import print_function
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
#import read_MITSceneParsingData as scene_parsing
import datetime
import time
#import BatchDatsetReader as dataset
from portrait import BatchDatset, ValDataset, TestDataset
from six.moves import xrange
from PIL import Image
from scipy import misc
import os
import tool
# from pspnet import Vgg19
os.chdir('/home/jrs1049/Desktop/QLMX/graduate')     #server
#os.chdir('/media/qlmx/Files/Graduate/project')        #linux

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/pspnet/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "data/fcn_vgg/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "model/fcn_vgg", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        # if name in ['conv3_4', 'relu3_4', 'conv4_4', 'relu4_4', 'conv5_4', 'relu5_4']:
        #     continue
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net

def pspnet(image, keep_prob):
    print("setting up pspnet initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]  # get image_mean
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])
    # processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, image)
        conv_final_layer = image_net["relu5_4"]
        print(conv_final_layer.shape)

        spp6 = tool.spatial_pyramid_pool(conv_final_layer)
        spp6_1x1_1 = tool.conv1x1_layer(spp6[0], 64, "spp6_1x1_1", True, kernel=1,
                                         relu=True, drop=True, alpha=0.2, batch=False)
        spp6_1x1_2 = tool.conv1x1_layer(spp6[1], 64, "spp6_1x1_2", True, kernel=1,
                                         relu=True, drop=True, alpha=0.2, batch=True)
        spp6_1x1_3 = tool.conv1x1_layer(spp6[2], 64, "spp6_1x1_3", True, kernel=1,
                                         relu=True, drop=True, alpha=0.2, batch=False)
        spp6_1x1_4 = tool.conv1x1_layer(spp6[3], 64, "spp6_1x1_4", True, kernel=1,
                                         relu=True, drop=True, alpha=0.2, batch=True)

        conv5_shape = conv_final_layer.get_shape().as_list()
        print(conv5_shape)
        upsample7_1 = tf.image.resize_bilinear(spp6_1x1_1, [conv5_shape[1], conv5_shape[2]])
        upsample7_2 = tf.image.resize_bilinear(spp6_1x1_2, [conv5_shape[1], conv5_shape[2]])
        upsample7_3 = tf.image.resize_bilinear(spp6_1x1_3, [conv5_shape[1], conv5_shape[2]])
        upsample7_4 = tf.image.resize_bilinear(spp6_1x1_4, [conv5_shape[1], conv5_shape[2]])

        conv1x1_8_1 = tool.conv1x1_layer(conv_final_layer, 256, "conv1x1_8_1", True, kernel=1,
                                          relu=True, drop=True, alpha=0.2, batch=True)
    # self.conv1x1_8_2 = self.conv1x1_layer(self.conv4_4, 256, "conv1x1_8_2", train_mode, kernel=1,
    #                                      relu=True, drop=True, alpha=0.2, batch=True)
        fuse8 = tf.concat([upsample7_1, upsample7_2, upsample7_3, upsample7_4, conv1x1_8_1],
                           axis=-1, name="fuse8")
        print(fuse8.shape)

        conv1x1_9 = tool.conv1x1_layer(fuse8, 512, "conv1x1_9", True, kernel=3,
                                        relu=True, drop=True, alpha=0.2, batch=True)
        print(conv1x1_9)

        convT_10 = tool.convT_layer(conv1x1_9, 128, "convT_10", True, relu=True, alpha=0.2)
        print(convT_10)

        convT_11 = tool.convT_layer(convT_10, 128, "convT_11", True, relu=True, alpha=0.2)
        print(convT_11)

        convT_12 = tool.convT_layer(convT_11, 64, "convT_12", True, relu=True, alpha=0.2)
    # self.conv1x1_12 = self.conv1x1_layer(self.convT_11, 64, "conv1x1_12", train_mode, relu=True, drop=False)
        print(convT_12)

        convT_12_2 = tool.convT_layer(convT_12, 64, "convT_13", True, relu=True, alpha=0.2)

        convT_13 = tool.conv1x1_layer(convT_12_2, 64, "convT_13", True, kernel=3,
                                       relu=True, drop=False, alpha=0.2, batch=False)
        print(convT_13)
        logits = tool.conv1x1_layer(convT_13, NUM_OF_CLASSESS, "logits", True, kernel=3,
                                     relu=False, drop=False, batch=False)

        preds = tf.nn.softmax(logits, name='preds')

        annotation_pred = tf.argmax(logits, axis=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), logits

def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob: the dropour rate.should have values in range 0-1.0
    :return: None
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)
    #model_data.key is dict_keys(['__header__', '__version__', '__globals__', 'layers', 'classes', 'normalization'])

    mean = model_data['normalization'][0][0][0]         #get image_mean
    mean_pixel = np.mean(mean, axis=(0, 1))


    weights = np.squeeze(model_data['layers'])
    #processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
        # print(relu_dropout7)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        annotation_pred1 = tf.argmax(conv8, axis=3, name="prediction1")
        # print(annotation_pred1)
        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
        # print(fuse_1)

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")
        # print(fuse_2)

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
        # print(pool5.shape)
        # print(relu6.shape, relu_dropout6.shape, relu7.shape, relu_dropout7.shape, conv8.shape, annotation_pred1.shape)
        # print(conv_t1.shape, fuse_1.shape, conv_t2.shape, fuse_2.shape, conv_t3.shape)

        annotation_pred = tf.argmax(conv_t3, axis=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    #config tensorflow gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")


    # pred_annotation, logits = pspnet(image, keep_probability)

    pred_annotation, logits = inference(image, keep_probability)
    print(logits.shape)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
    tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up data reader...")
    train_dataset_reader = BatchDatset('data/fcn_vgg/trainlist.mat')
    validation_dataset_reader = ValDataset('data/fcn_vgg/testlist.mat')
    test_dataset_reader = TestDataset('data/test/testlist.mat')

    sess = tf.Session(config=config)

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    #start to train
    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch()
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.5}
            #train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            #print('==> batch data: ', train_images[0][100][100], '===', train_annotations[0][100][100])

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 20 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            #if itr % 100 == 0:
            #    rpred = sess.run(pred_annotation, feed_dict=feed_dict)
            #    print(np.sum(rpred))
            #    print('=============')
            #    print(np.sum(train_annotations))
            #    print('------------>>>')

            if itr % 500 == 0:
                valid_images, valid_annotations, valid_orgs = validation_dataset_reader.next_batch()
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
def pred():
	keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
	image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="input_image")
	annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")	

	pred_annotation, logits = inference(image, keep_probability)
	test_dataset_reader = TestDataset('data/fcn_vgg/testlist.mat')
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
		saver = tf.train.Saver()
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print("Model restored...")

		test_images, test_annotations, test_orgs = test_dataset_reader.next_batch()
		itr = 0
		while len(test_images) > 0:
			feed_dict = {image: test_images, annotation: test_annotations, keep_probability: 0.5}
			preds = sess.run(pred_annotation, feed_dict=feed_dict)
			org0_im = Image.fromarray(np.uint8(test_orgs[0]))
			org0_im.save('result/fcn_vgg2/org%d.jpg' % itr)
			org1_im = Image.fromarray(np.uint8(test_orgs[1]))
			org1_im.save('result/fcn_vgg2/org%d.jpg' % (itr + 1))
			save_alpha_img(test_orgs[0], test_annotations[0], 'result/fcn_vgg2/ann%d'%itr)
			save_alpha_img(test_orgs[1], test_annotations[1], 'result/fcn_vgg2/ann%d'%(itr+1))
			save_alpha_img(test_orgs[0], preds[0], 'result/fcn_vgg2/pre%d' % itr)
			save_alpha_img(test_orgs[1], preds[1], 'result/fcn_vgg2/pre%d' % (itr + 1))
			test_images, test_annotations, test_orgs = test_dataset_reader.next_batch()
			itr += 2

def save_alpha_img(org, mat, name):
    w, h, _ = mat.shape
    #print(mat[200:210, 200:210])
    rmat = np.reshape(mat, (w, h))
    amat = np.zeros((w, h, 4), dtype=np.int)
    amat[:, :, 3] = rmat * 1000
    amat[:, :, 0:3] = org
    print(amat[200:205, 200:205])
    #im = Image.fromarray(np.uint8(amat))
    #im.save(name + '.png')
    misc.imsave(name + '.png', amat)


# def save_alpha_img(org, mat, name):
#     w, h, _ = mat.shape
#
#     # #print(mat[200:210, 200:210])
#     rmat = np.reshape(mat, (w, h))
#     amat = np.zeros((w, h, 3), dtype=np.int)
#     print(mat.shape)
#     print(amat.shape)
#     for i in range(3):
#         amat[:, :, i] = rmat*255
#     # # amat[:, :, 0:3] = org
#     # # print(amat[200:205, 200:205])
#     # #im = Image.fromarray(np.uint8(amat))
#     # #im.save(name + '.png')
#     misc.imsave(name + '.png', amat)


if __name__ == "__main__":
    main()
    # pred()
